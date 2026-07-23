//! Persistent, content-addressed embedding cache. Vectors are keyed by
//! `embedding_key` (a hash of the ranking text bound to the ingest fingerprint,
//! excluding the line span), so an identical chunk embedded under the same
//! pipeline — on a `--reindex`, or in *another* project that fetched the same
//! vendor doc — is reused instead of recomputed. The store lives under
//! `$XDG_CACHE_HOME/rag-bone/` (cross-project) and is keyed per `(model, dim)`
//! so vectors are never mixed across incompatible pipelines. It is best-effort:
//! a load or save error never fails an index build. Disable with
//! `RAG_NO_EMB_CACHE=1`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

const CACHE_MAGIC: &[u8; 8] = b"RAGBEMB\0";

#[derive(Serialize, Deserialize)]
struct Persisted {
    dim: usize,
    entries: HashMap<u64, Vec<f32>>,
}

/// In-memory embedding cache for one `(model, dim)`, backed by an on-disk file.
#[derive(Default)]
pub struct EmbeddingCache {
    dim: usize,
    path: Option<PathBuf>,
    entries: HashMap<u64, Vec<f32>>,
    dirty: bool,
}

impl EmbeddingCache {
    /// An empty, non-persistent cache (used when caching is disabled).
    pub fn empty(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }

    /// Load the cache for `(model, dim)`, or return an empty one. A missing,
    /// unreadable, corrupt, or dimension-mismatched file yields an empty cache —
    /// caching never fails a build.
    pub fn load(model: &str, dim: usize) -> Self {
        let Some(path) = cache_path(model, dim) else {
            return Self::empty(dim);
        };
        let entries = read_entries(&path, dim).unwrap_or_default();
        debug!(cached = entries.len(), path = %path.display(), "embedding cache loaded");
        Self {
            dim,
            path: Some(path),
            entries,
            dirty: false,
        }
    }

    pub fn get(&self, key: u64) -> Option<&[f32]> {
        self.entries.get(&key).map(Vec::as_slice)
    }

    pub fn insert(&mut self, key: u64, vector: Vec<f32>) {
        if vector.len() == self.dim {
            self.entries.insert(key, vector);
            self.dirty = true;
        }
    }

    /// Persist to disk if changed. Best-effort: a write error is logged, not
    /// propagated (a warm cache is an optimization, never a correctness input).
    pub fn save(&self) {
        let (Some(path), true) = (self.path.as_ref(), self.dirty) else {
            return;
        };
        if let Err(error) = write_entries(path, self.dim, &self.entries) {
            warn!(%error, path = %path.display(), "could not persist embedding cache");
        }
    }
}

/// Resolve embeddings for the parallel `keys`/`texts` arrays: serve hits from
/// `cache`, embed only the misses through `embed` (called once, in input order),
/// splice the results back into position, and write new vectors into the cache.
/// Returns `(vectors, hits, misses)`.
pub fn resolve_with_cache<F>(
    cache: &mut EmbeddingCache,
    keys: &[u64],
    texts: &[String],
    embed: F,
) -> Result<(Vec<Vec<f32>>, usize, usize)>
where
    F: FnOnce(&[String]) -> Result<Vec<Vec<f32>>>,
{
    debug_assert_eq!(keys.len(), texts.len());
    let mut out: Vec<Option<Vec<f32>>> = vec![None; keys.len()];
    let mut miss_positions: Vec<usize> = Vec::new();
    let mut miss_texts: Vec<String> = Vec::new();
    for (i, &key) in keys.iter().enumerate() {
        match cache.get(key) {
            Some(vector) => out[i] = Some(vector.to_vec()),
            None => {
                miss_positions.push(i);
                miss_texts.push(texts[i].clone());
            }
        }
    }
    let hits = keys.len() - miss_positions.len();
    let misses = miss_positions.len();
    if !miss_texts.is_empty() {
        let embedded = embed(&miss_texts)?;
        anyhow::ensure!(
            embedded.len() == miss_texts.len(),
            "embedder returned {} vectors for {} texts",
            embedded.len(),
            miss_texts.len()
        );
        for (pos, vector) in miss_positions.into_iter().zip(embedded) {
            cache.insert(keys[pos], vector.clone());
            out[pos] = Some(vector);
        }
    }
    let vectors = out
        .into_iter()
        .map(|v| v.expect("every position resolved"))
        .collect();
    Ok((vectors, hits, misses))
}

/// `$XDG_CACHE_HOME/rag-bone/emb-<model>-<dim>.postcard`, falling back to
/// `$HOME/.cache`. `None` if neither environment variable is set.
fn cache_path(model: &str, dim: usize) -> Option<PathBuf> {
    let root = crate::config::cache_root()?;
    let safe_model: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    Some(root.join(format!("emb-{safe_model}-{dim}.postcard")))
}

fn read_entries(path: &Path, dim: usize) -> Result<HashMap<u64, Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let body = bytes
        .strip_prefix(CACHE_MAGIC.as_slice())
        .ok_or_else(|| anyhow::anyhow!("{} is not a rag-bone embedding cache", path.display()))?;
    let persisted: Persisted = postcard::from_bytes(body)?;
    anyhow::ensure!(
        persisted.dim == dim,
        "embedding cache dim {} does not match {dim}",
        persisted.dim
    );
    // Defensive: drop any entry whose vector length disagrees with the header.
    Ok(persisted
        .entries
        .into_iter()
        .filter(|(_, vector)| vector.len() == dim)
        .collect())
}

fn write_entries(path: &Path, dim: usize, entries: &HashMap<u64, Vec<f32>>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let persisted = Persisted {
        dim,
        entries: entries.clone(),
    };
    let mut bytes = CACHE_MAGIC.to_vec();
    bytes.extend(postcard::to_allocvec(&persisted)?);
    // Write to a temporary sibling, then rename over the target.
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_uses_cache_and_embeds_only_misses_in_order() {
        let mut cache = EmbeddingCache::empty(2);
        cache.insert(10, vec![1.0, 0.0]); // pre-seed the first key
        let keys = [10u64, 20, 30];
        let texts = ["a".to_string(), "b".to_string(), "c".to_string()];
        let mut embedded_texts: Vec<String> = Vec::new();
        let (vectors, hits, misses) = resolve_with_cache(&mut cache, &keys, &texts, |miss| {
            embedded_texts = miss.to_vec();
            Ok(miss.iter().map(|_| vec![0.5, 0.5]).collect())
        })
        .unwrap();
        assert_eq!((hits, misses), (1, 2));
        assert_eq!(embedded_texts, ["b", "c"]); // only misses, in input order
        assert_eq!(vectors[0], vec![1.0, 0.0]); // served from cache
        assert_eq!(vectors[1], vec![0.5, 0.5]);
        assert_eq!(vectors[2], vec![0.5, 0.5]);
        assert_eq!(cache.get(20), Some([0.5, 0.5].as_slice())); // written back
    }

    #[test]
    fn resolve_all_hits_never_calls_embed() {
        let mut cache = EmbeddingCache::empty(1);
        cache.insert(1, vec![9.0]);
        let (vectors, hits, misses) =
            resolve_with_cache(&mut cache, &[1], &["x".to_string()], |_| {
                panic!("embed must not run when everything is cached")
            })
            .unwrap();
        assert_eq!((hits, misses), (1, 0));
        assert_eq!(vectors, vec![vec![9.0]]);
    }

    #[test]
    fn roundtrip_persists_and_rejects_dim_mismatch() {
        let dir = std::env::temp_dir().join(format!("ragbone-emb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("emb.postcard");
        let mut entries = HashMap::new();
        entries.insert(7u64, vec![1.0f32, 2.0]);
        write_entries(&path, 2, &entries).unwrap();
        assert_eq!(
            read_entries(&path, 2).unwrap().get(&7),
            Some(&vec![1.0, 2.0])
        );
        assert!(read_entries(&path, 3).is_err()); // dim mismatch → caller falls back
    }
}
