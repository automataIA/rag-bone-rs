use crate::config::ChunkConfig;
use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// A selectable embedding model plus the retrieval prefixes it expects.
/// Prefixes matter: asymmetric models (nomic, bge, embeddinggemma) are trained
/// with distinct query/document instructions and lose accuracy without them.
pub struct ModelSpec {
    pub id: &'static str,
    pub model: EmbeddingModel,
    pub query_prefix: &'static str,
    pub doc_prefix: &'static str,
}

/// Shortlist for the Fase B autoresearch sweep (all ONNX, CPU-friendly, no giants).
const REGISTRY: &[ModelSpec] = &[
    ModelSpec {
        id: "nomic-embed-text-v1.5",
        model: EmbeddingModel::NomicEmbedTextV15,
        query_prefix: "search_query: ",
        doc_prefix: "search_document: ",
    },
    ModelSpec {
        id: "jina-embeddings-v2-base-code",
        model: EmbeddingModel::JinaEmbeddingsV2BaseCode,
        query_prefix: "",
        doc_prefix: "",
    },
    ModelSpec {
        id: "embeddinggemma-300m",
        model: EmbeddingModel::EmbeddingGemma300M,
        query_prefix: "task: search result | query: ",
        doc_prefix: "title: none | text: ",
    },
    ModelSpec {
        id: "gte-base-en-v1.5",
        model: EmbeddingModel::GTEBaseENV15,
        query_prefix: "",
        doc_prefix: "",
    },
    ModelSpec {
        id: "bge-base-en-v1.5",
        model: EmbeddingModel::BGEBaseENV15,
        query_prefix: "Represent this sentence for searching relevant passages: ",
        doc_prefix: "",
    },
    ModelSpec {
        id: "bge-small-en-v1.5",
        model: EmbeddingModel::BGESmallENV15,
        query_prefix: "Represent this sentence for searching relevant passages: ",
        doc_prefix: "",
    },
    // Quantized (int8) variants: ~half the RAM/disk for ~1-2 points of accuracy.
    // Same retrieval prefixes as their fp32 counterparts.
    ModelSpec {
        id: "nomic-embed-text-v1.5-q",
        model: EmbeddingModel::NomicEmbedTextV15Q,
        query_prefix: "search_query: ",
        doc_prefix: "search_document: ",
    },
    ModelSpec {
        id: "embeddinggemma-300m-q",
        model: EmbeddingModel::EmbeddingGemma300MQ,
        query_prefix: "task: search result | query: ",
        doc_prefix: "title: none | text: ",
    },
    // int4 weight-only (QuantizationMode::None, so batch-safe unlike the dynamic
    // `-q` above): the quantized embeddinggemma that actually indexes here.
    ModelSpec {
        id: "embeddinggemma-300m-q4",
        model: EmbeddingModel::EmbeddingGemma300MQ4,
        query_prefix: "task: search result | query: ",
        doc_prefix: "title: none | text: ",
    },
    ModelSpec {
        id: "gte-base-en-v1.5-q",
        model: EmbeddingModel::GTEBaseENV15Q,
        query_prefix: "",
        doc_prefix: "",
    },
    ModelSpec {
        id: "bge-base-en-v1.5-q",
        model: EmbeddingModel::BGEBaseENV15Q,
        query_prefix: "Represent this sentence for searching relevant passages: ",
        doc_prefix: "",
    },
    ModelSpec {
        id: "bge-small-en-v1.5-q",
        model: EmbeddingModel::BGESmallENV15Q,
        query_prefix: "Represent this sentence for searching relevant passages: ",
        doc_prefix: "",
    },
];

/// Token cap per chunk. A 1500-char chunk is well under this for code/prose, so
/// there is no truncation in practice — but it bounds padding, which is what
/// keeps peak memory in check (some models default to an 8192-token max).
const MAX_TOKENS: usize = 512;
/// Documents embedded per forward pass. Small batches keep the padded activation
/// tensors (batch × seq × hidden) from blowing up RAM.
const EMBED_BATCH: usize = 16;
/// Bump whenever stored document embeddings can change without a model id or
/// chunk configuration change (for example a new document prefix or chunker).
/// v2: documents are embedded as `ranking_text` (structural header + raw text)
/// instead of raw text alone.
const INGEST_PIPELINE_VERSION: u32 = 2;

fn spec(id: &str) -> Result<&'static ModelSpec> {
    REGISTRY.iter().find(|s| s.id == id).ok_or_else(|| {
        let ids = REGISTRY.iter().map(|s| s.id).collect::<Vec<_>>().join(", ");
        anyhow::anyhow!("unknown model '{id}'; available: {ids}")
    })
}

/// Loaded embedding model. Produces L2-normalized vectors so cosine similarity
/// reduces to a dot product downstream.
pub struct Embedder {
    inner: TextEmbedding,
    spec: &'static ModelSpec,
    dim: usize,
}

impl Embedder {
    pub fn load(id: &str) -> Result<Self> {
        let spec = spec(id)?;
        let dim = TextEmbedding::get_model_info(&spec.model)
            .with_context(|| format!("model info for '{id}'"))?
            .dim;
        let mut opts = InitOptions::new(spec.model.clone())
            .with_max_length(MAX_TOKENS)
            .with_execution_providers(crate::accel::execution_providers());
        // Keep the multi-GB weights in one cross-project cache instead of a
        // `.fastembed_cache` beside whatever directory the binary runs in.
        if let Some(dir) = crate::config::model_cache_dir() {
            opts = opts.with_cache_dir(dir);
        }
        let inner =
            TextEmbedding::try_new(opts).with_context(|| format!("loading model '{id}'"))?;
        Ok(Self { inner, spec, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Stable fingerprint of every setting that affects persisted chunks or vectors.
    pub fn ingest_fingerprint(&self, chunk: &ChunkConfig) -> u64 {
        ingest_fingerprint(self.spec, self.dim, chunk)
    }

    /// Embed documents for storage (applies the doc prefix).
    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.spec.doc_prefix))
            .collect();
        let mut vecs = self
            .inner
            .embed(prefixed, Some(EMBED_BATCH))
            .context("embedding documents")?;
        vecs.iter_mut().for_each(|v| normalize(v));
        Ok(vecs)
    }

    /// Embed a single query (applies the query prefix).
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let text = format!("{}{query}", self.spec.query_prefix);
        let mut vecs = self
            .inner
            .embed(vec![text], None)
            .context("embedding query")?;
        let mut v = vecs.pop().context("empty embedding result")?;
        normalize(&mut v);
        Ok(v)
    }
}

fn ingest_fingerprint(spec: &ModelSpec, dim: usize, chunk: &ChunkConfig) -> u64 {
    let descriptor = format!(
        "v={INGEST_PIPELINE_VERSION}\0model={}\0doc_prefix={}\0dim={dim}\0max_tokens={MAX_TOKENS}\0max_chars={}\0overlap={}",
        spec.id, spec.doc_prefix, chunk.max_chars, chunk.overlap
    );
    xxhash_rust::xxh3::xxh3_64(descriptor.as_bytes())
}

/// L2-normalize in place; leaves an all-zero vector untouched.
fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lookup() {
        assert!(spec("nomic-embed-text-v1.5").is_ok());
        assert!(spec("does-not-exist").is_err());
    }

    #[test]
    fn normalize_unit_norm() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ingest_fingerprint_tracks_chunk_config() {
        let spec = spec("embeddinggemma-300m-q4").unwrap();
        let base = ChunkConfig {
            max_chars: 1500,
            overlap: 0,
        };
        let changed = ChunkConfig {
            max_chars: 1200,
            overlap: 0,
        };
        assert_ne!(
            ingest_fingerprint(spec, 768, &base),
            ingest_fingerprint(spec, 768, &changed)
        );
    }
}
