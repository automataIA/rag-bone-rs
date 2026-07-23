use anyhow::{Context, Result};
use fastembed::{
    OnnxSource, RerankInitOptions, RerankInitOptionsUserDefined, RerankerModel, TextRerank,
    TokenizerFiles, UserDefinedRerankingModel,
};
use std::path::Path;

/// Token cap and batch size — same memory-bounding rationale as the embedder.
const MAX_TOKENS: usize = 512;
const RERANK_BATCH: usize = 16;

/// A selectable cross-encoder reranker (all fastembed built-ins, ONNX/CPU-or-GPU).
struct RerankerSpec {
    id: &'static str,
    model: RerankerModel,
}

/// Built-in reranker registry. The application default is the lighter
/// auto-downloaded MiniLM recipe; these remain selectable quality/multilingual
/// alternatives (and may have different licensing terms).
const REGISTRY: &[RerankerSpec] = &[
    RerankerSpec {
        id: "bge-reranker-base",
        model: RerankerModel::BGERerankerBase,
    },
    RerankerSpec {
        id: "jina-reranker-v1-turbo-en",
        model: RerankerModel::JINARerankerV1TurboEn,
    },
    RerankerSpec {
        id: "bge-reranker-v2-m3",
        model: RerankerModel::BGERerankerV2M3,
    },
    RerankerSpec {
        id: "jina-reranker-v2-base-multilingual",
        model: RerankerModel::JINARerankerV2BaseMultiligual,
    },
];

/// Cross-encoder reranker. Scores each (query, document) pair jointly, which is
/// far more precise than bi-encoder cosine — the second stage that turns broad
/// recall into a few high-quality results.
pub struct Reranker {
    inner: TextRerank,
}

impl Reranker {
    /// Load a reranker. `id` resolution order:
    /// 1. a `.onnx` file path → bring-your-own ONNX;
    /// 2. a built-in registry id → fastembed auto-download;
    /// 3. a known HF recipe (e.g. `ms-marco-MiniLM-L6-v2`), using a local CPU
    ///    prefetch when appropriate or auto-downloading a provider-specific ONNX;
    /// 4. another local prefetch at `<root>/models/rerank/<id>/model.onnx`;
    ///    `hf-hub` into the shared cache (zero-setup, like the built-ins).
    pub fn load(id: &str, root: &Path) -> Result<Self> {
        let models_dir = root.join("models").join("rerank");
        let configured_path = if Path::new(id).is_absolute() {
            Path::new(id).to_path_buf()
        } else {
            root.join(id)
        };
        let inner = if id.ends_with(".onnx") || configured_path.is_file() {
            load_user_defined(&configured_path)?
        } else if let Some(spec) = REGISTRY.iter().find(|s| s.id == id) {
            let mut opts = RerankInitOptions::new(spec.model.clone())
                .with_max_length(MAX_TOKENS)
                .with_execution_providers(crate::accel::execution_providers());
            if let Some(dir) = crate::config::model_cache_dir() {
                opts = opts.with_cache_dir(dir);
            }
            TextRerank::try_new(opts).with_context(|| format!("loading reranker '{id}'"))?
        } else if let Some(recipe) = HF_RERANKERS.iter().find(|r| r.name == id) {
            let model_dir = models_dir.join(id);
            let provider_prefetch = model_dir.join("model-accelerated.onnx");
            let cpu_prefetch = model_dir.join("model.onnx");
            if crate::accel::prefers_fp32_onnx() && provider_prefetch.is_file() {
                load_user_defined(&provider_prefetch)?
            } else if !crate::accel::prefers_fp32_onnx() && cpu_prefetch.is_file() {
                load_user_defined(&cpu_prefetch)?
            } else {
                fetch_hf_reranker(recipe)?
            }
        } else if models_dir.join(id).join("model.onnx").is_file() {
            load_user_defined(&models_dir.join(id).join("model.onnx"))?
        } else {
            let builtin = REGISTRY.iter().map(|s| s.id).collect::<Vec<_>>().join(", ");
            let hf = HF_RERANKERS
                .iter()
                .map(|r| r.name)
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "unknown reranker '{id}'; built-in: {builtin}; auto-download: {hf}; \
                 or a path to a .onnx / a prefetch under {}",
                models_dir.display()
            );
        };
        Ok(Self { inner })
    }

    /// Rerank `documents` against `query`, returning `(original_index, score)`
    /// pairs ordered best-first. Raw cross-encoder logits are squashed with a
    /// sigmoid into (0, 1) so they are usable with a model-specific `min_score`
    /// threshold. Scores are not calibrated across different rerankers.
    pub fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let docs: Vec<&str> = documents.iter().map(String::as_str).collect();
        let results = self
            .inner
            .rerank(query, &docs, false, Some(RERANK_BATCH))
            .context("reranking")?;
        Ok(results
            .into_iter()
            .map(|r| (r.index, sigmoid(r.score)))
            .collect())
    }
}

/// A user-defined reranker fetchable from the HF Hub. `onnx_file` picks an
/// arch-specific int8 export on CPU or FP32 for an accelerated provider.
struct HfReranker {
    name: &'static str,
    repo: &'static str,
    onnx_file: fn() -> &'static str,
}

/// Recipes auto-downloadable via `hf-hub` (zero-setup, cached like the built-ins).
/// All are Apache-2.0 cross-encoders with self-contained provider-specific ONNX
/// exports and standard tokenizer files, validated against the HF repos.
const HF_RERANKERS: &[HfReranker] = &[
    // EN, BERT cross-encoders. CPU uses arch-specific int8 (~23-34 MB), while
    // accelerated builds use FP32. L6 = default/fastest; L12 has more layers.
    HfReranker {
        name: "ms-marco-MiniLM-L6-v2",
        repo: "cross-encoder/ms-marco-MiniLM-L6-v2",
        onnx_file: ms_marco_onnx_for_provider,
    },
    HfReranker {
        name: "ms-marco-MiniLM-L12-v2",
        repo: "cross-encoder/ms-marco-MiniLM-L12-v2",
        onnx_file: ms_marco_onnx_for_provider,
    },
    // EN long-context (ModernBERT, 8192 tok, 149M) — strong on docs + code.
    HfReranker {
        name: "gte-reranker-modernbert-base",
        repo: "Alibaba-NLP/gte-reranker-modernbert-base",
        onnx_file: gte_onnx_for_provider,
    },
    // Multilingual (70+ langs, 306M, 8192 tok) — commercial-friendly (Apache-2.0).
    HfReranker {
        name: "gte-multilingual-reranker-base",
        repo: "onnx-community/gte-multilingual-reranker-base",
        onnx_file: gte_onnx_for_provider,
    },
];

/// Accelerated providers use FP32; CPU picks the int8 ONNX matching host SIMD.
/// L6 and L12 share identical in-repo filenames, so these functions serve both.
#[cfg(any(feature = "gpu", feature = "directml", feature = "coreml"))]
fn ms_marco_onnx_for_provider() -> &'static str {
    "onnx/model.onnx"
}

#[cfg(not(any(feature = "gpu", feature = "directml", feature = "coreml")))]
fn ms_marco_onnx_for_provider() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512vnni") {
            return "onnx/model_qint8_avx512_vnni.onnx";
        }
        if std::is_x86_feature_detected!("avx512f") {
            return "onnx/model_qint8_avx512.onnx";
        }
        if std::is_x86_feature_detected!("avx2") {
            return "onnx/model_quint8_avx2.onnx";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "onnx/model_qint8_arm64.onnx";
    }
    "onnx/model.onnx" // fp32 fallback
}

#[cfg(any(feature = "gpu", feature = "directml", feature = "coreml"))]
fn gte_onnx_for_provider() -> &'static str {
    "onnx/model.onnx"
}

/// Single-file int8 ONNX for CPU recipes without arch-specific variants.
#[cfg(not(any(feature = "gpu", feature = "directml", feature = "coreml")))]
fn gte_onnx_for_provider() -> &'static str {
    "onnx/model_int8.onnx"
}

/// Auto-download a recipe's ONNX + tokenizer files into the shared HF cache
/// (`~/.cache/huggingface`) and build the reranker. Cached after the first run.
fn fetch_hf_reranker(recipe: &HfReranker) -> Result<TextRerank> {
    use hf_hub::api::sync::ApiBuilder;
    let repo = ApiBuilder::new()
        .build()
        .context("initializing hf-hub api")?
        .model(recipe.repo.to_string());
    let get = |file: &str| {
        repo.get(file)
            .with_context(|| format!("downloading {file} from {}", recipe.repo))
    };
    let onnx = get((recipe.onnx_file)())?;
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(get("tokenizer.json")?)?,
        config_file: std::fs::read(get("config.json")?)?,
        special_tokens_map_file: std::fs::read(get("special_tokens_map.json")?)?,
        tokenizer_config_file: std::fs::read(get("tokenizer_config.json")?)?,
    };
    build_reranker(OnnxSource::File(onnx), tokenizer_files)
}

/// Load a "bring your own" ONNX cross-encoder from a local `.onnx` file. The four
/// tokenizer JSONs are read from the model's own directory; external weights
/// (`.onnx_data`) are resolved by ORT relative to the `.onnx` path.
fn load_user_defined(onnx_path: &Path) -> Result<TextRerank> {
    let dir = onnx_path.parent().unwrap_or_else(|| Path::new("."));
    let read = |name: &str| {
        std::fs::read(dir.join(name))
            .with_context(|| format!("reading {name} for reranker {}", onnx_path.display()))
    };
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };
    build_reranker(OnnxSource::File(onnx_path.to_path_buf()), tokenizer_files)
}

/// Build a `TextRerank` from a user-defined ONNX source and tokenizer bytes.
fn build_reranker(onnx: OnnxSource, tokenizer_files: TokenizerFiles) -> Result<TextRerank> {
    let model = UserDefinedRerankingModel::new(onnx, tokenizer_files);
    let opts = RerankInitOptionsUserDefined::new()
        .with_max_length(MAX_TOKENS)
        .with_execution_providers(crate::accel::execution_providers());
    TextRerank::try_new_from_user_defined(model, opts).context("loading user-defined reranker")
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_export_matches_execution_provider() {
        if crate::accel::prefers_fp32_onnx() {
            assert_eq!(ms_marco_onnx_for_provider(), "onnx/model.onnx");
            assert_eq!(gte_onnx_for_provider(), "onnx/model.onnx");
        } else {
            assert!(ms_marco_onnx_for_provider().starts_with("onnx/model_"));
            assert_eq!(gte_onnx_for_provider(), "onnx/model_int8.onnx");
        }
    }
}
