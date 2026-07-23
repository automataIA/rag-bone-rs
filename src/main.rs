mod accel;
mod chunk;
mod cli;
mod config;
mod embcache;
mod embed;
mod eval;
mod index;
mod lexical;
mod metadata;
mod output;
mod progress;
mod rerank;
mod search;
mod store;
mod walk;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command};
use config::Config;
use embed::Embedder;
use rerank::Reranker;
use search::SearchParams;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use store::Index;
use tracing::info;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let config_path = cli
        .config
        .clone()
        .unwrap_or_else(|| PathBuf::from(config::CONFIG_FILE));

    match cli.command {
        Command::Config { init } => cmd_config(&config_path, init),
        Command::Index {
            path,
            model,
            reindex,
        } => cmd_index(&config_path, path, model, reindex),
        Command::Search {
            query,
            limit,
            json,
            compact,
            no_rerank,
            reranker,
            retrieval,
            lang,
            path_prefix,
            source,
        } => cmd_search(
            &config_path,
            query,
            limit,
            json,
            compact,
            no_rerank,
            reranker,
            retrieval,
            lang,
            path_prefix,
            source,
        ),
        Command::SearchBatch {
            limit,
            compact,
            no_rerank,
            reranker,
            retrieval,
            lang,
            path_prefix,
            source,
        } => cmd_search_batch(
            &config_path,
            limit,
            compact,
            no_rerank,
            reranker,
            retrieval,
            lang,
            path_prefix,
            source,
        ),
        Command::Get {
            chunk_id,
            context_lines,
            json,
        } => cmd_get(&config_path, &chunk_id, context_lines, json),
        Command::Outline { file, json } => cmd_outline(&config_path, &file, json),
        Command::Find {
            symbol,
            limit,
            json,
        } => cmd_find(&config_path, &symbol, limit, json),
        Command::Catalog { json } => cmd_catalog(&config_path, json),
        Command::Status => cmd_status(&config_path),
        Command::Watch { path } => cmd_watch(&config_path, path),
        Command::Eval {
            queries,
            k,
            no_rerank,
            reranker,
            retrieval,
            dump,
        } => cmd_eval(
            &config_path,
            &queries,
            k,
            no_rerank,
            reranker,
            retrieval,
            dump,
        ),
    }
}

/// Build the derived BM25 channel when the resolved retrieval mode needs it.
/// `RAG_LEXICAL_RAW=1` excludes the structural header from the lexical document
/// (the raw-vs-ranking_text ablation axis of Fase 3); it defaults to included.
fn build_lexical(
    index: &Index,
    mode: config::RetrievalMode,
) -> Result<Option<lexical::LexicalIndex>> {
    if mode == config::RetrievalMode::Dense {
        return Ok(None);
    }
    let include_structural = std::env::var_os("RAG_LEXICAL_RAW").is_none();
    info!(include_structural, "building lexical (BM25) channel");
    Ok(Some(lexical::LexicalIndex::build(
        &index.records,
        include_structural,
    )?))
}

fn cmd_config(config_path: &Path, init: bool) -> Result<()> {
    if init {
        Config::init(config_path)?;
        println!("Wrote {}", config_path.display());
    } else {
        let cfg = Config::load(config_path)?;
        print!("{}", toml::to_string_pretty(&cfg)?);
    }
    Ok(())
}

fn cmd_index(
    config_path: &Path,
    path: Vec<PathBuf>,
    model: Option<String>,
    reindex: bool,
) -> Result<()> {
    let mut cfg = Config::load(config_path)?;
    if !path.is_empty() {
        cfg.sources = path;
    }
    if let Some(m) = model {
        cfg.model = m;
    }
    let root = root_of(config_path);
    let ipath = index_path(&root);

    info!(model = %cfg.model, "loading embedding model");
    let mut embedder = Embedder::load(&cfg.model)?;
    let (_, stats) = index::build(&root, &cfg, &ipath, &mut embedder, reindex)?;
    println!(
        "indexed {} files ({} chunks), skipped {}, failed {}, pruned {} → {}",
        stats.files_indexed,
        stats.chunks,
        stats.files_skipped,
        stats.files_failed,
        stats.files_pruned,
        ipath.display()
    );
    if stats.cache_hits + stats.cache_misses > 0 {
        println!(
            "embedding cache: {} hits, {} misses",
            stats.cache_hits, stats.cache_misses
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_search(
    config_path: &Path,
    query: String,
    limit: Option<usize>,
    json: bool,
    compact: bool,
    no_rerank: bool,
    reranker: Option<String>,
    retrieval: Option<config::RetrievalMode>,
    lang: Option<Vec<String>>,
    path_prefix: Option<String>,
    source: Option<String>,
) -> Result<()> {
    let cfg = Config::load(config_path)?;
    let root = root_of(config_path);
    let index =
        Index::load(&index_path(&root)).context("no index found — run `rag-bone index` first")?;
    if index.is_empty() {
        anyhow::bail!("index is empty — run `rag-bone index` first");
    }

    let mode = retrieval.unwrap_or(cfg.retrieval);
    let mut embedder = if mode == config::RetrievalMode::Bm25 {
        None
    } else {
        info!(model = %index.model, "loading embedding model");
        Some(Embedder::load(&index.model)?)
    };
    let do_rerank = cfg.rerank && !no_rerank;
    let reranker_id = reranker.as_deref().unwrap_or(&cfg.reranker);
    let mut reranker = do_rerank
        .then(|| {
            info!(reranker = reranker_id, "loading reranker");
            Reranker::load(reranker_id, &root)
        })
        .transpose()?;

    let top_k = limit.unwrap_or(cfg.top_k);
    anyhow::ensure!(top_k > 0, "--limit must be greater than zero");

    let lexical = build_lexical(&index, mode)?;
    let params = SearchParams {
        query,
        top_k,
        retrieve_n: cfg.retrieve_n.max(top_k),
        rerank: do_rerank,
        min_score: cfg.min_score,
        langs: lang,
        path_prefix,
        source,
        retrieval: mode,
        rrf_k: cfg.rrf_k,
        max_per_file: cfg.max_per_file,
    };
    let results = search::run(
        &index,
        embedder.as_mut(),
        reranker.as_mut(),
        lexical.as_ref(),
        &params,
    )?;
    output::print_results(&results, json, compact)
}

#[allow(clippy::too_many_arguments)]
fn cmd_search_batch(
    config_path: &Path,
    limit: Option<usize>,
    compact: bool,
    no_rerank: bool,
    reranker: Option<String>,
    retrieval: Option<config::RetrievalMode>,
    lang: Option<Vec<String>>,
    path_prefix: Option<String>,
    source: Option<String>,
) -> Result<()> {
    let cfg = Config::load(config_path)?;
    let root = root_of(config_path);
    let index =
        Index::load(&index_path(&root)).context("no index found — run `rag-bone index` first")?;
    anyhow::ensure!(
        !index.is_empty(),
        "index is empty — run `rag-bone index` first"
    );

    let top_k = limit.unwrap_or(cfg.top_k);
    anyhow::ensure!(top_k > 0, "--limit must be greater than zero");
    let mode = retrieval.unwrap_or(cfg.retrieval);
    let mut embedder = if mode == config::RetrievalMode::Bm25 {
        None
    } else {
        info!(model = %index.model, "loading embedding model");
        Some(Embedder::load(&index.model)?)
    };
    let do_rerank = cfg.rerank && !no_rerank;
    let reranker_id = reranker.as_deref().unwrap_or(&cfg.reranker);
    let mut reranker = do_rerank
        .then(|| {
            info!(reranker = reranker_id, "loading reranker");
            Reranker::load(reranker_id, &root)
        })
        .transpose()?;

    let lexical = build_lexical(&index, mode)?;
    let stdin = std::io::stdin();
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let query = line.context("reading a query from stdin")?;
        let query = query.trim();
        if query.is_empty() {
            continue;
        }
        let params = SearchParams {
            query: query.to_string(),
            top_k,
            retrieve_n: cfg.retrieve_n.max(top_k),
            rerank: do_rerank,
            min_score: cfg.min_score,
            langs: lang.clone(),
            path_prefix: path_prefix.clone(),
            source: source.clone(),
            retrieval: mode,
            rrf_k: cfg.rrf_k,
            max_per_file: cfg.max_per_file,
        };
        let results = search::run(
            &index,
            embedder.as_mut(),
            reranker.as_mut(),
            lexical.as_ref(),
            &params,
        )?;
        output::write_jsonl_record(&mut stdout, query, &results, compact)?;
    }
    Ok(())
}

/// Fetch one chunk by `chunk_id`, loading no models. Distinguishes a missing/
/// corrupt index (load error) from an unknown id (index loads but has no match),
/// each with an actionable message. `--context-lines N` widens the span by
/// reading the source file; if the file is gone, it falls back to the stored
/// snippet and warns on stderr.
fn cmd_get(
    config_path: &Path,
    chunk_id: &str,
    context_lines: Option<usize>,
    json: bool,
) -> Result<()> {
    let root = root_of(config_path);
    let index = Index::load(&index_path(&root))
        .context("no readable index — run `rag-bone index` first (or `--reindex` if corrupt)")?;
    let record = index.get_by_chunk_id(chunk_id).ok_or_else(|| {
        anyhow::anyhow!(
            "chunk id '{chunk_id}' not found in the index — list ids with `rag-bone search --compact`"
        )
    })?;

    // Default to the stored raw text; widen from the file only when asked.
    let (lines, text) = match context_lines {
        Some(n) if n > 0 => read_with_context(&root, record, n),
        _ => ([record.start_line, record.end_line], record.text.clone()),
    };

    if json {
        let value = serde_json::json!({
            "chunk_id": record.chunk_id,
            "file": record.file,
            "lang": record.lang,
            "symbol": record.symbol,
            "kind": record.kind,
            "headings": record.headings,
            "source": record.source,
            "corpus_source": record.corpus_source,
            "fetched": record.fetched,
            "lines": lines,
            "text": text,
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        println!("{}:{}-{}", record.file.display(), lines[0], lines[1]);
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
    }
    Ok(())
}

/// Read `[start-n, end+n]` (1-based, clamped) from the record's source file. On a
/// read error, warn and fall back to the stored span/snippet — `get` still works
/// against a stale index, it just cannot widen the context.
fn read_with_context(root: &Path, record: &store::Record, n: usize) -> ([usize; 2], String) {
    let path = root.join(&record.file);
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let file_lines: Vec<&str> = content.lines().collect();
            let start = record.start_line.saturating_sub(n).max(1);
            let end = (record.end_line + n).min(file_lines.len());
            let slice = file_lines[start - 1..end].join("\n");
            ([start, end], slice)
        }
        Err(error) => {
            tracing::warn!(file = %path.display(), %error, "cannot read source for context; using stored snippet");
            ([record.start_line, record.end_line], record.text.clone())
        }
    }
}

/// Structural outline of a file — its indexed chunks with symbol/kind/span — with
/// no models loaded. Errors if the path matches no indexed chunk.
fn cmd_outline(config_path: &Path, file: &str, json: bool) -> Result<()> {
    let root = root_of(config_path);
    let index = Index::load(&index_path(&root))
        .context("no readable index — run `rag-bone index` first (or `--reindex` if corrupt)")?;
    let records = index.outline(file);
    anyhow::ensure!(
        !records.is_empty(),
        "no indexed chunk matches '{file}' — check the path or run `rag-bone index`"
    );
    output::print_record_meta(&records, json)
}

/// Jump-to-definition for a symbol name, models-free. Primary: index records whose
/// `symbol` field matches (exact first, then substring). Fallback: when no symbol
/// metadata matches — common when a chunk straddles a definition boundary, so its
/// symbol degraded to `None` — the derived BM25 channel surfaces lexical candidates,
/// reordered so records that *define* the name rank above call-sites/tests. Errors
/// only if both come up empty.
fn cmd_find(config_path: &Path, symbol: &str, limit: Option<usize>, json: bool) -> Result<()> {
    let root = root_of(config_path);
    let index = Index::load(&index_path(&root))
        .context("no readable index — run `rag-bone index` first (or `--reindex` if corrupt)")?;
    let limit = limit.unwrap_or(10);

    let mut records = index.find_symbol(symbol);
    if records.is_empty() {
        info!(
            symbol,
            "no symbol metadata match; falling back to lexical candidates"
        );
        let lexical = lexical::LexicalIndex::build(&index.records, true)?;
        // Pull a wider BM25 pool, then float records that *define* the symbol (a
        // def keyword precedes the name on a line) above call-sites/tests, which
        // pure term-frequency otherwise ranks first. Measured on 165 fallback
        // symbols: MRR 0.628→0.797, recall@1 52%→78%, +0.7 ms/call, 0 regressions.
        let mut cands = lexical.search(symbol, (limit * 4).max(20), |_| true);
        // Stable partition: definition-like first, BM25 order preserved within.
        cands.sort_by_key(|&(idx, _)| !is_definition_like(&index.records[idx].text, symbol));
        records = cands
            .into_iter()
            .map(|(idx, _)| &index.records[idx])
            .collect();
    }
    anyhow::ensure!(
        !records.is_empty(),
        "no symbol or lexical match for '{symbol}' — try a broader semantic search: `rag-bone search \"{symbol}\"`"
    );
    records.truncate(limit);
    output::print_record_meta(&records, json)
}

/// Whether `text` contains a *definition* of `name` (not merely a mention): some
/// line has a definition keyword token before the `name` token. Shared across
/// languages; deliberately excludes binding keywords (`let`/`const`/`var`), whose
/// RHS is usually a call site, to avoid ranking `let x = name()` as a definition.
fn is_definition_like(text: &str, name: &str) -> bool {
    const DEF_KW: &[&str] = &[
        "fn",
        "def",
        "func",
        "function",
        "class",
        "struct",
        "enum",
        "trait",
        "interface",
        "type",
        "impl",
        "mod",
        "module",
        "package",
    ];
    let name_lower = name.to_lowercase();
    text.lines().any(|line| {
        let tokens: Vec<&str> = line
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|t| !t.is_empty())
            .collect();
        let Some(name_pos) = tokens.iter().position(|t| t.to_lowercase() == name_lower) else {
            return false;
        };
        tokens[..name_pos]
            .iter()
            .any(|t| DEF_KW.contains(&t.to_ascii_lowercase().as_str()))
    })
}

/// List the web-sourced document sets in the index (models-free): what external
/// docs are indexed and how old, so an agent can scope a query with `--source`.
fn cmd_catalog(config_path: &Path, json: bool) -> Result<()> {
    let root = root_of(config_path);
    let index = Index::load(&index_path(&root))
        .context("no readable index — run `rag-bone index` first (or `--reindex` if corrupt)")?;
    output::print_catalog(&index.catalog(), json)
}

fn cmd_status(config_path: &Path) -> Result<()> {
    let root = root_of(config_path);
    let ipath = index_path(&root);
    let index = Index::load(&ipath).context("no index found — run `rag-bone index` first")?;
    println!("model:    {}", index.model);
    println!("dim:      {}", index.dim);
    println!("files:    {}", index.file_count());
    println!("chunks:   {}", index.len());
    println!("provider: {}", accel::provider_name());
    println!("index:    {}", ipath.display());
    Ok(())
}

fn cmd_watch(config_path: &Path, path: Vec<PathBuf>) -> Result<()> {
    use notify::{RecursiveMode, Watcher};
    use std::sync::mpsc;
    use std::time::Duration;

    let mut cfg = Config::load(config_path)?;
    if !path.is_empty() {
        cfg.sources = path;
    }
    let root = root_of(config_path);
    let ipath = index_path(&root);

    let mut embedder = Embedder::load(&cfg.model)?;
    let (_, stats) = index::build(&root, &cfg, &ipath, &mut embedder, false)?;
    info!(
        files = stats.files_indexed,
        chunks = stats.chunks,
        "initial index built"
    );

    let (tx, rx) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;
    for src in &cfg.sources {
        let dir = root.join(src);
        if dir.exists() {
            watcher.watch(&dir, RecursiveMode::Recursive)?;
            info!(dir = %dir.display(), "watching");
        }
    }

    loop {
        if rx.recv().is_err() {
            break;
        }
        // Debounce a burst of filesystem events into one rebuild.
        std::thread::sleep(Duration::from_millis(500));
        while rx.try_recv().is_ok() {}
        let (_, stats) = index::build(&root, &cfg, &ipath, &mut embedder, false)?;
        info!(
            indexed = stats.files_indexed,
            pruned = stats.files_pruned,
            chunks = stats.chunks,
            "reindexed"
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_eval(
    config_path: &Path,
    queries: &Path,
    k: usize,
    no_rerank: bool,
    reranker: Option<String>,
    retrieval: Option<config::RetrievalMode>,
    dump: Option<PathBuf>,
) -> Result<()> {
    anyhow::ensure!(k > 0, "--k must be greater than zero");
    let cfg = Config::load(config_path)?;
    let root = root_of(config_path);
    let index =
        Index::load(&index_path(&root)).context("no index found — run `rag-bone index` first")?;
    if index.is_empty() {
        anyhow::bail!("index is empty — run `rag-bone index` first");
    }

    info!(model = %index.model, "loading embedding model");
    let mut embedder = Embedder::load(&index.model)?;
    let do_rerank = cfg.rerank && !no_rerank;
    let reranker_id = reranker.as_deref().unwrap_or(&cfg.reranker);
    let mut reranker = do_rerank
        .then(|| {
            info!(reranker = reranker_id, "loading reranker");
            Reranker::load(reranker_id, &root)
        })
        .transpose()?;

    let mode = retrieval.unwrap_or(cfg.retrieval);
    let lexical = build_lexical(&index, mode)?;
    let opts = eval::EvalOpts {
        k,
        retrieve_n: cfg.retrieve_n,
        rerank: do_rerank,
        min_score: cfg.min_score,
        retrieval: mode,
        rrf_k: cfg.rrf_k,
        dump,
    };
    let report = eval::run(
        &index,
        &mut embedder,
        reranker.as_mut(),
        lexical.as_ref(),
        queries,
        &opts,
    )?;

    // Single tsv-friendly line on stdout for easy capture; details on stderr.
    info!(
        model = %index.model,
        rerank = do_rerank,
        retrieve_n = cfg.retrieve_n,
        min_score = cfg.min_score,
        "evaluated {} queries",
        report.n
    );
    let span_str = |span: Option<(f32, usize)>| match span {
        Some((rate, n)) => format!("   Span@{} = {rate:.3} (n={n})", report.k),
        None => String::new(),
    };
    println!(
        "Recall@{} = {:.3}   Recall@10 = {:.3}   MRR = {:.3}   nDCG@10 = {:.3}{}   {:.0} ms/query   ({} queries)",
        report.k,
        report.recall_at_k,
        report.recall_at_10,
        report.mrr,
        report.ndcg_at_10,
        span_str(report.span),
        report.mean_latency_ms,
        report.n
    );
    // Per-stratum breakdown. Small strata are noisy — the query count per row is
    // printed so a one-query flip in a tiny stratum is not mistaken for a signal.
    for s in &report.strata {
        println!(
            "  [{:<16}] Recall@{} = {:.3}   Recall@10 = {:.3}   MRR = {:.3}   nDCG@10 = {:.3}{}   ({} queries)",
            s.category,
            report.k,
            s.recall_at_k,
            s.recall_at_10,
            s.mrr,
            s.ndcg_at_10,
            span_str(s.span),
            s.n
        );
    }
    Ok(())
}

/// Directory containing the config file — the root for relative sources and the index.
fn root_of(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn index_path(root: &Path) -> PathBuf {
    root.join(config::INDEX_DIR).join("index.bin")
}

#[cfg(test)]
mod tests {
    use super::is_definition_like;

    #[test]
    fn definition_line_is_recognized_across_languages() {
        assert!(is_definition_like(
            "pub fn tokenize(text: &str) {",
            "tokenize"
        ));
        assert!(is_definition_like("    def parse(self):", "parse"));
        assert!(is_definition_like("func Add(a int) int {", "Add"));
        assert!(is_definition_like("struct Point {", "Point"));
        assert!(is_definition_like("impl Index {", "Index"));
    }

    #[test]
    fn call_sites_and_bindings_are_not_definitions() {
        // A call, no def keyword before the name.
        assert!(!is_definition_like(
            "    let v = tokenize(query);",
            "tokenize"
        ));
        assert!(!is_definition_like(
            "        self.tokenize(&doc)",
            "tokenize"
        ));
        // Name absent entirely.
        assert!(!is_definition_like("fn other() {}", "tokenize"));
    }
}
