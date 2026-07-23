use crate::config::RetrievalMode;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// High-performance semantic search over docs and code.
#[derive(Parser)]
#[command(name = "rag-bone", version, about)]
pub struct Cli {
    /// Path to the config file (default: ./.rag-bone.toml).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Ingest the configured folders into the index.
    Index {
        /// Override the config `sources` (repeatable).
        #[arg(long)]
        path: Vec<PathBuf>,
        /// Override the embedding model.
        #[arg(long)]
        model: Option<String>,
        /// Rebuild from scratch, ignoring cached hashes.
        #[arg(long)]
        reindex: bool,
    },
    /// Semantic search; prints the most relevant chunks.
    Search {
        /// The natural-language query.
        query: String,
        /// Max results to return.
        #[arg(short = 'k', long)]
        limit: Option<usize>,
        /// Emit a JSON array on stdout (for agents).
        #[arg(long)]
        json: bool,
        /// Compact output: chunk_id + one-line preview instead of full snippets.
        #[arg(long)]
        compact: bool,
        /// Disable the cross-encoder rerank stage.
        #[arg(long)]
        no_rerank: bool,
        /// Override the reranker (registry id, model name, or path to a .onnx).
        #[arg(long)]
        reranker: Option<String>,
        /// Retrieval channel override (dense, bm25, hybrid).
        #[arg(long)]
        retrieval: Option<RetrievalMode>,
        /// Restrict to languages (comma-separated, e.g. rust,md).
        #[arg(long, value_delimiter = ',')]
        lang: Option<Vec<String>>,
        /// Keep only results whose path contains this substring.
        #[arg(long)]
        path_prefix: Option<String>,
        /// Scope to web docs whose source URL contains this substring (see `catalog`).
        #[arg(long)]
        source: Option<String>,
    },
    /// Search one query per stdin line, keeping both models loaded; emits JSONL.
    SearchBatch {
        /// Max results to return for each query.
        #[arg(short = 'k', long)]
        limit: Option<usize>,
        /// Compact JSONL: chunk_id + one-line preview instead of full snippets.
        #[arg(long)]
        compact: bool,
        /// Disable the cross-encoder rerank stage.
        #[arg(long)]
        no_rerank: bool,
        /// Override the reranker (registry id, model name, or path to a .onnx).
        #[arg(long)]
        reranker: Option<String>,
        /// Retrieval channel override (dense, bm25, hybrid).
        #[arg(long)]
        retrieval: Option<RetrievalMode>,
        /// Restrict to languages (comma-separated, e.g. rust,md).
        #[arg(long, value_delimiter = ',')]
        lang: Option<Vec<String>>,
        /// Keep only results whose path contains this substring.
        #[arg(long)]
        path_prefix: Option<String>,
        /// Scope to web docs whose source URL contains this substring (see `catalog`).
        #[arg(long)]
        source: Option<String>,
    },
    /// Watch the configured folders and reindex on change.
    Watch {
        #[arg(long)]
        path: Vec<PathBuf>,
    },
    /// Evaluate a golden set in-process (Recall@k, MRR, mean latency).
    Eval {
        /// Golden set tsv: `query <TAB> expected_path_substring` (`|`-separated alts).
        #[arg(long, default_value = "eval/queries.tsv")]
        queries: PathBuf,
        /// Cutoff for Recall@k (also the number of results scored).
        #[arg(short = 'k', long, default_value_t = 3)]
        k: usize,
        /// Disable the cross-encoder rerank stage.
        #[arg(long)]
        no_rerank: bool,
        /// Override the reranker (registry id, model name, or path to a .onnx).
        #[arg(long)]
        reranker: Option<String>,
        /// Retrieval channel override (dense, bm25, hybrid).
        #[arg(long)]
        retrieval: Option<RetrievalMode>,
        /// Write per-query top-k results to this JSONL file (for the LLM judge).
        #[arg(long)]
        dump: Option<PathBuf>,
    },
    /// Fetch one chunk by its `chunk_id` (from `search --compact`). Loads no
    /// models, so it is fast. With `--context-lines`, expands the span by reading
    /// the source file.
    Get {
        /// The chunk_id to fetch.
        chunk_id: String,
        /// Include N source lines above and below the chunk's span (reads the file).
        #[arg(long)]
        context_lines: Option<usize>,
        /// Emit a JSON object on stdout (for agents).
        #[arg(long)]
        json: bool,
    },
    /// Structural outline of a file: its indexed chunks with symbol, kind, span
    /// and signature, in source order. Loads no models.
    Outline {
        /// File path (or a path substring) to outline.
        file: String,
        /// Emit a JSON array on stdout (for agents).
        #[arg(long)]
        json: bool,
    },
    /// Find where a symbol is defined: index records whose symbol matches `name`
    /// (exact first, then substring). Loads no models.
    Find {
        /// Symbol name to look up.
        symbol: String,
        /// Max results to return.
        #[arg(short = 'k', long)]
        limit: Option<usize>,
        /// Emit a JSON array on stdout (for agents).
        #[arg(long)]
        json: bool,
    },
    /// List web-sourced document sets in the index (source URL, fetch date,
    /// file/chunk counts) — the agent's routing menu before `search --source`.
    /// Loads no models.
    Catalog {
        /// Emit a JSON array on stdout (for agents).
        #[arg(long)]
        json: bool,
    },
    /// Show index statistics (chunks, files, model, provider).
    Status,
    /// Create or print the config file.
    Config {
        /// Write a default .rag-bone.toml.
        #[arg(long)]
        init: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_name_matches_installed_binary() {
        let error = match Cli::try_parse_from(["rag-bone", "--version"]) {
            Ok(_) => panic!("--version should exit through clap"),
            Err(error) => error,
        };
        assert_eq!(error.exit_code(), 0);
        assert!(error.to_string().starts_with("rag-bone "));
    }

    #[test]
    fn parses_search_batch_filters() {
        let cli = Cli::try_parse_from([
            "rag-bone",
            "search-batch",
            "--limit",
            "2",
            "--lang",
            "rust,md",
            "--path-prefix",
            "src/",
        ])
        .unwrap();
        let Command::SearchBatch {
            limit,
            lang,
            path_prefix,
            ..
        } = cli.command
        else {
            panic!("expected search-batch command");
        };
        assert_eq!(limit, Some(2));
        assert_eq!(lang.unwrap(), ["rust", "md"]);
        assert_eq!(path_prefix.as_deref(), Some("src/"));
    }
}
