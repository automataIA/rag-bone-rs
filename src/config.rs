use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Config file at the root of the project being indexed.
pub const CONFIG_FILE: &str = ".rag-bone.toml";
/// Directory holding the persisted index, relative to the project root.
pub const INDEX_DIR: &str = ".rag-bone";

/// Root of rag-bone's cross-project cache: `$XDG_CACHE_HOME/rag-bone`, else
/// `$HOME/.cache/rag-bone`. `None` when neither variable is set, in which case
/// callers fall back to their own local default.
pub fn cache_root() -> Option<PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .map(|base| base.join("rag-bone"))
}

/// Where ONNX model weights are cached. Global on purpose: the weights are
/// multiple GB and identical across projects, so fastembed's default (a
/// `.fastembed_cache` beside the current directory) would re-download the whole
/// set the first time the binary runs in a new project. Override with
/// `RAG_BONE_MODEL_CACHE`; note that `HF_HOME`, if set, still wins inside
/// fastembed itself.
pub fn model_cache_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("RAG_BONE_MODEL_CACHE") {
        return Some(PathBuf::from(dir));
    }
    cache_root().map(|root| root.join("models"))
}
/// Default embedding model, chosen by the Fase B autoresearch sweep: the int4
/// weight-only embeddinggemma (batch-safe, unlike the dynamic `-q`) leads on every
/// quality metric (Recall@3 0.978 / MRR 0.893 / judge 0.964) at ~6s/query and a
/// light footprint. bge-base-en-v1.5-q is the int8 runner-up. See `docs/RESEARCH.md`.
pub const DEFAULT_MODEL: &str = "embeddinggemma-300m-q4";

/// Retrieval channel. `dense` is the exact-cosine baseline; `bm25` is the
/// code-aware lexical channel alone; `hybrid` fuses both with Reciprocal Rank
/// Fusion. The lexical channel is derived in-memory from the loaded index, so it
/// carries no persisted artifact and cannot drift out of sync with the vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
#[clap(rename_all = "lower")]
pub enum RetrievalMode {
    #[default]
    Dense,
    Bm25,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Folders to ingest. Relative paths resolve against the config location.
    pub sources: Vec<PathBuf>,
    /// Embedding model id (see `embed::registry`).
    pub model: String,
    /// Enable the cross-encoder rerank stage.
    pub rerank: bool,
    /// Cross-encoder reranker: a built-in registry id, a named user-defined model
    /// under `models/rerank/<name>/`, or a path to a `.onnx`.
    pub reranker: String,
    /// Results returned to the caller.
    pub top_k: usize,
    /// Candidates pulled from vector search before reranking.
    pub retrieve_n: usize,
    /// Drop final results below this score (0.0 = keep all).
    pub min_score: f32,
    /// Retrieval channel: dense, bm25, or hybrid (dense + BM25 via RRF).
    pub retrieval: RetrievalMode,
    /// Reciprocal Rank Fusion constant `k` (hybrid only). A hyperparameter, not a
    /// universal truth; ~60 is the common default. Higher flattens rank influence.
    pub rrf_k: usize,
    /// Max chunks returned per file, applied after ranking to diversify results
    /// (0 = unlimited). Guards against three near-duplicate chunks of one file
    /// crowding out other sources.
    pub max_per_file: usize,
    pub chunk: ChunkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkConfig {
    /// Chunk size budget, measured in characters by `text-splitter`.
    pub max_chars: usize,
    /// Overlap between adjacent chunks, in characters.
    pub overlap: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sources: vec![PathBuf::from("./docs"), PathBuf::from("./src")],
            model: DEFAULT_MODEL.to_string(),
            // Fase 3 ablation: on the eval corpus the cross-encoder demotes the
            // right chunk of a near-perfect hybrid candidate set (and costs ~50×),
            // so the default pipeline is hybrid without the reranker. It stays
            // selectable (`rerank = true`, `--reranker`) for corpora where it helps.
            rerank: false,
            reranker: "ms-marco-MiniLM-L6-v2".to_string(),
            top_k: 3,
            // Sweep: Recall@3/MRR flat across 30/50/100, so pick the fastest.
            retrieve_n: 30,
            min_score: 0.0,
            // Fase 3: dense + BM25 fused with RRF beats dense alone on every
            // metric (Recall@3 1.000, MRR 0.963, Span@3 0.545) at ~18 ms/query.
            retrieval: RetrievalMode::Hybrid,
            rrf_k: 60,
            max_per_file: 0,
            chunk: ChunkConfig::default(),
        }
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chars: 1500,
            overlap: 0,
        }
    }
}

impl Config {
    /// Load from `path`, or return defaults if the file does not exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            let config = Self::default();
            config.validate()?;
            return Ok(config);
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let config: Self =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        config
            .validate()
            .with_context(|| format!("validating {}", path.display()))?;
        Ok(config)
    }

    /// Serialize defaults to `path`, refusing to clobber an existing file.
    pub fn init(path: &Path) -> Result<()> {
        if path.exists() {
            anyhow::bail!("{} already exists", path.display());
        }
        let text = toml::to_string_pretty(&Self::default()).context("serializing config")?;
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.model.trim().is_empty(), "model must not be empty");
        anyhow::ensure!(
            !self.rerank || !self.reranker.trim().is_empty(),
            "reranker must not be empty when rerank is enabled"
        );
        anyhow::ensure!(self.top_k > 0, "top_k must be greater than zero");
        anyhow::ensure!(self.retrieve_n > 0, "retrieve_n must be greater than zero");
        anyhow::ensure!(self.rrf_k > 0, "rrf_k must be greater than zero");
        anyhow::ensure!(self.min_score.is_finite(), "min_score must be finite");
        anyhow::ensure!(
            self.chunk.max_chars > 0,
            "chunk.max_chars must be greater than zero"
        );
        anyhow::ensure!(
            self.chunk.overlap < self.chunk.max_chars,
            "chunk.overlap must be smaller than chunk.max_chars"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_roundtrip() {
        let text = toml::to_string_pretty(&Config::default()).unwrap();
        let parsed: Config = toml::from_str(&text).unwrap();
        assert_eq!(parsed.model, DEFAULT_MODEL);
        assert!(!parsed.rerank);
        assert_eq!(parsed.retrieval, RetrievalMode::Hybrid);
        assert_eq!(parsed.top_k, 3);
        assert_eq!(parsed.chunk.max_chars, 1500);
    }

    #[test]
    fn partial_config_fills_defaults() {
        let parsed: Config = toml::from_str("top_k = 5\n").unwrap();
        assert_eq!(parsed.top_k, 5);
        assert_eq!(parsed.retrieve_n, 30);
        assert_eq!(parsed.model, DEFAULT_MODEL);
    }

    #[test]
    fn load_missing_returns_default() {
        let cfg = Config::load(Path::new("/nonexistent/.rag-bone.toml")).unwrap();
        assert_eq!(cfg.top_k, 3);
    }

    #[test]
    fn rejects_invalid_values() {
        let mut cfg = Config::default();
        cfg.chunk.overlap = cfg.chunk.max_chars;
        assert!(cfg.validate().is_err());

        let cfg = Config {
            top_k: 0,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());

        let cfg = Config {
            min_score: f32::NAN,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }
}
