use crate::config::RetrievalMode;
use crate::embed::Embedder;
use crate::lexical::LexicalIndex;
use crate::rerank::Reranker;
use crate::store::Index;
use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// One returned result: enough for an agent to open the exact span, or to fetch
/// it later by `chunk_id` with `get`. `symbol`/`kind` are omitted for prose.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub file: PathBuf,
    pub lang: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub headings: Vec<String>,
    /// Origin URL and fetch date for web-sourced chunks (from frontmatter);
    /// omitted for local files. Lets the agent cite the source and its age.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corpus_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fetched: Option<String>,
    pub lines: [usize; 2],
    pub score: f32,
    pub snippet: String,
}

/// Query-time knobs. Defaults come from `Config`, flags may override per call.
pub struct SearchParams {
    pub query: String,
    pub top_k: usize,
    pub retrieve_n: usize,
    pub rerank: bool,
    pub min_score: f32,
    pub langs: Option<Vec<String>>,
    pub path_prefix: Option<String>,
    /// Keep only web-sourced results whose `source` URL contains this substring
    /// (scopes a query to one external doc set); local files never match.
    pub source: Option<String>,
    pub retrieval: RetrievalMode,
    pub rrf_k: usize,
    /// Max chunks kept per file after ranking (0 = unlimited).
    pub max_per_file: usize,
}

/// Run the full pipeline: retrieve `retrieve_n` candidates on the configured
/// channel(s) with metadata filters applied during the scan → optional rerank →
/// keep `top_k` above `min_score`. `lexical` must be present for the `bm25`/
/// `hybrid` channels; a missing lexical channel is a hard error, never a silent
/// fall back to dense (that would falsify a hybrid benchmark).
pub fn run(
    index: &Index,
    embedder: Option<&mut Embedder>,
    reranker: Option<&mut Reranker>,
    lexical: Option<&LexicalIndex>,
    params: &SearchParams,
) -> Result<Vec<SearchResult>> {
    let candidates = retrieve(index, embedder, lexical, params)?;

    let ranked = match reranker {
        Some(r) if params.rerank => {
            let docs: Vec<String> = candidates
                .iter()
                .map(|&(idx, _)| index.records[idx].text.clone())
                .collect();
            let mut ranked = Vec::with_capacity(docs.len());
            for (doc_pos, score) in r.rerank(&params.query, &docs)? {
                let candidate = candidates.get(doc_pos).ok_or_else(|| {
                    anyhow::anyhow!(
                        "reranker returned candidate index {doc_pos}, but only {} candidates exist",
                        candidates.len()
                    )
                })?;
                ranked.push((candidate.0, score));
            }
            ranked
        }
        _ => candidates,
    };

    // Apply the score threshold, then optionally diversify by capping how many
    // chunks may come from one file (so three near-duplicate chunks of the same
    // file don't crowd out other sources), then keep the final top_k.
    let mut per_file: HashMap<&std::path::Path, usize> = HashMap::new();
    Ok(ranked
        .into_iter()
        .filter(|&(_, score)| passes_min_score(score, params.min_score))
        .filter(|&(idx, _)| {
            within_file_cap(&mut per_file, &index.records[idx].file, params.max_per_file)
        })
        .take(params.top_k)
        .map(|(idx, score)| {
            let r = &index.records[idx];
            SearchResult {
                chunk_id: r.chunk_id.clone(),
                file: r.file.clone(),
                lang: r.lang.clone(),
                symbol: r.symbol.clone(),
                kind: r.kind.clone(),
                headings: r.headings.clone(),
                source: r.source.clone(),
                corpus_source: r.corpus_source.clone(),
                fetched: r.fetched.clone(),
                lines: [r.start_line, r.end_line],
                score,
                snippet: r.text.clone(),
            }
        })
        .collect())
}

/// Diversification predicate: with `cap == 0` every result passes; otherwise a
/// file may contribute at most `cap` chunks. Stateful — call in ranked order.
fn within_file_cap<'a>(
    per_file: &mut HashMap<&'a std::path::Path, usize>,
    file: &'a std::path::Path,
    cap: usize,
) -> bool {
    if cap == 0 {
        return true;
    }
    let count = per_file.entry(file).or_insert(0);
    if *count < cap {
        *count += 1;
        true
    } else {
        false
    }
}

/// Retrieve up to `retrieve_n` filtered candidates on the configured channel.
/// For `hybrid`, dense and lexical are retrieved separately, fused with RRF, and
/// the union is cut back to `retrieve_n` *before* the reranker — so the
/// cross-encoder never sees more candidates than the dense-only path would, and
/// its cost (the dominant CPU term) does not roughly double.
fn retrieve(
    index: &Index,
    embedder: Option<&mut Embedder>,
    lexical: Option<&LexicalIndex>,
    params: &SearchParams,
) -> Result<Vec<(usize, f32)>> {
    let dense = |embedder: &mut Embedder| -> Result<Vec<(usize, f32)>> {
        let query_vec = embedder.embed_query(&params.query)?;
        Ok(index
            .search_filtered(&query_vec, params.retrieve_n, |record| {
                passes_filters(record, params)
            })?
            .into_iter()
            .map(|h| (h.idx, h.score))
            .collect())
    };
    // BM25 filters on the same predicate, keyed by record index.
    let bm25 = |lexical: &LexicalIndex| {
        lexical.search(&params.query, params.retrieve_n, |idx| {
            passes_filters(&index.records[idx], params)
        })
    };
    // `bm25`/`hybrid` require the derived lexical channel; its absence is a bug
    // in the caller, surfaced loudly rather than silently downgrading to dense.
    let need_lexical = || {
        lexical.ok_or_else(|| {
            anyhow::anyhow!(
                "retrieval mode requires the lexical channel, which was not built \
                 — rebuild the index with `rag-bone index --reindex`"
            )
        })
    };

    let need_embedder = || {
        embedder.ok_or_else(|| {
            anyhow::anyhow!("retrieval mode requires the embedding model, but it was not loaded")
        })
    };

    match params.retrieval {
        RetrievalMode::Dense => dense(need_embedder()?),
        RetrievalMode::Bm25 => Ok(bm25(need_lexical()?)),
        RetrievalMode::Hybrid => {
            let lexical = need_lexical()?;
            let dense = dense(need_embedder()?)?;
            let lexical = bm25(lexical);
            Ok(rrf_fuse(&dense, &lexical, params.rrf_k, params.retrieve_n))
        }
    }
}

/// Reciprocal Rank Fusion of two ranked lists, keyed by record index (each record
/// is one `chunk_id`, so the map deduplicates the union by chunk). A document's
/// fused score is `Σ 1/(k + rank)` over the lists it appears in (rank 1-based).
/// The union is sorted best-first and truncated to `cap`.
fn rrf_fuse(
    dense: &[(usize, f32)],
    lexical: &[(usize, f32)],
    k: usize,
    cap: usize,
) -> Vec<(usize, f32)> {
    let mut fused: HashMap<usize, f32> = HashMap::new();
    for list in [dense, lexical] {
        for (rank, &(idx, _)) in list.iter().enumerate() {
            *fused.entry(idx).or_insert(0.0) += 1.0 / (k + rank + 1) as f32;
        }
    }
    let mut ranked: Vec<(usize, f32)> = fused.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(cap);
    ranked
}

fn passes_filters(record: &crate::store::Record, params: &SearchParams) -> bool {
    let lang_ok = params
        .langs
        .as_ref()
        .is_none_or(|langs| langs.iter().any(|l| l == &record.lang));
    let path_ok = params
        .path_prefix
        .as_ref()
        .is_none_or(|p| record.file.to_string_lossy().contains(p.as_str()));
    let source_ok = params.source.as_ref().is_none_or(|s| {
        [record.source.as_deref(), record.corpus_source.as_deref()]
            .into_iter()
            .flatten()
            .any(|src| src.contains(s.as_str()))
    });
    lang_ok && path_ok && source_ok
}

fn passes_min_score(score: f32, min_score: f32) -> bool {
    min_score == 0.0 || score >= min_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Record;

    fn params() -> SearchParams {
        SearchParams {
            query: "q".into(),
            top_k: 3,
            retrieve_n: 50,
            rerank: false,
            min_score: 0.0,
            langs: None,
            path_prefix: None,
            source: None,
            retrieval: RetrievalMode::Dense,
            rrf_k: 60,
            max_per_file: 0,
        }
    }

    fn record(file: &str, lang: &str) -> Record {
        Record {
            file: PathBuf::from(file),
            lang: lang.into(),
            start_line: 1,
            end_line: 2,
            chunk_id: format!("1-{file}"),
            symbol: None,
            kind: None,
            parent: None,
            signature: None,
            headings: Vec::new(),
            source: None,
            corpus_source: None,
            fetched: None,
            embedding_key: 0,
            text: "x".into(),
            vector: vec![1.0, 0.0],
        }
    }

    #[test]
    fn lang_filter_matches() {
        let mut p = params();
        p.langs = Some(vec!["rust".into()]);
        assert!(passes_filters(&record("a.rs", "rust"), &p));
        assert!(!passes_filters(&record("a.md", "md"), &p));
    }

    #[test]
    fn path_prefix_matches() {
        let mut p = params();
        p.path_prefix = Some("src/".into());
        assert!(passes_filters(&record("src/a.rs", "rust"), &p));
        assert!(!passes_filters(&record("docs/a.rs", "rust"), &p));
    }

    #[test]
    fn source_filter_matches_web_docs_only() {
        let mut p = params();
        p.source = Some("docs.rs".into());
        let mut web = record("vendor/serde.md", "md");
        web.source = Some("https://docs.rs/serde".into());
        assert!(passes_filters(&web, &p));
        let mut other = record("vendor/tokio.md", "md");
        other.source = Some("https://tokio.rs".into());
        assert!(!passes_filters(&other, &p));
        // A local file (no source) never matches a source filter.
        assert!(!passes_filters(&record("src/a.rs", "rust"), &p));

        let mut corpus = record("vendor/full.md", "md");
        corpus.source = Some("https://site.test/requested-page".into());
        corpus.corpus_source = Some("https://site.test/llms-full.txt".into());
        p.source = Some("llms-full".into());
        assert!(passes_filters(&corpus, &p));
    }

    #[test]
    fn zero_min_score_disables_filtering() {
        assert!(passes_min_score(-0.25, 0.0));
        assert!(!passes_min_score(-0.25, 0.1));
        assert!(passes_min_score(0.5, 0.1));
    }

    #[test]
    fn rrf_dedups_union_and_rewards_agreement() {
        // Record 1 is top of both lists → highest fused score despite neither
        // channel's raw score; record 0 and 2 appear once each.
        let dense = vec![(0usize, 0.9f32), (1, 0.8)];
        let lexical = vec![(1usize, 5.0f32), (2, 4.0)];
        let fused = rrf_fuse(&dense, &lexical, 60, 10);
        assert_eq!(fused.len(), 3, "union deduplicates the shared record");
        assert_eq!(fused[0].0, 1, "the record both channels rank should win");
        // Fused score for record 1: 1/(60+2) + 1/(60+1); for record 0: 1/(60+1).
        let expect_1 = 1.0 / 62.0 + 1.0 / 61.0;
        assert!((fused[0].1 - expect_1).abs() < 1e-6);
    }

    #[test]
    fn within_file_cap_limits_per_file_but_zero_is_unlimited() {
        use std::path::Path;
        let a = Path::new("src/a.rs");
        let b = Path::new("src/b.rs");
        // cap 0 → everything passes.
        let mut seen = std::collections::HashMap::new();
        assert!(within_file_cap(&mut seen, a, 0));
        assert!(within_file_cap(&mut seen, a, 0));
        // cap 1 → the second chunk of the same file is dropped, other files pass.
        let mut seen = std::collections::HashMap::new();
        assert!(within_file_cap(&mut seen, a, 1));
        assert!(!within_file_cap(&mut seen, a, 1));
        assert!(within_file_cap(&mut seen, b, 1));
    }

    #[test]
    fn rrf_truncates_union_to_cap_before_rerank() {
        let dense = vec![(0usize, 0.9f32), (1, 0.8), (2, 0.7)];
        let lexical = vec![(3usize, 5.0f32), (4, 4.0), (5, 3.0)];
        // Six distinct records, but the cap keeps only two for the reranker.
        let fused = rrf_fuse(&dense, &lexical, 60, 2);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn bm25_runs_without_an_embedder() {
        let mut index = Index::new("model-that-must-not-load", 2, 1);
        let mut first = record("docs/first.md", "md");
        first.text = "alpha retrieval contract".into();
        let mut second = record("docs/second.md", "md");
        second.text = "unrelated material".into();
        index.records = vec![first, second];
        let lexical = LexicalIndex::build(&index.records, true).unwrap();
        let mut p = params();
        p.query = "alpha contract".into();
        p.retrieval = RetrievalMode::Bm25;

        let results = run(&index, None, None, Some(&lexical), &p).unwrap();
        assert_eq!(results[0].file, PathBuf::from("docs/first.md"));
    }

    #[test]
    fn dense_fails_loudly_without_an_embedder() {
        let index = Index::new("missing", 2, 1);
        let error = run(&index, None, None, None, &params()).unwrap_err();
        assert!(error.to_string().contains("embedding model"));
    }
}
