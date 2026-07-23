use crate::chunk::{self, Lang};
use crate::config::Config;
use crate::embcache::{self, EmbeddingCache};
use crate::embed::Embedder;
use crate::metadata;
use crate::progress::Progress;
use crate::store::{Index, Record};
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Summary of an ingest run, reported to the user.
#[derive(Debug, Default)]
pub struct Stats {
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_pruned: usize,
    pub files_failed: usize,
    pub chunks: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Build or update the index for `root` using `config`. Unchanged files (same
/// content hash) are skipped; deleted files are pruned. A model change forces a
/// full rebuild. The ingestion fingerprint also covers chunking and document
/// prefix settings, which affect record boundaries or vector contents.
pub fn build(
    root: &Path,
    config: &Config,
    index_path: &Path,
    embedder: &mut Embedder,
    reindex: bool,
) -> Result<(Index, Stats)> {
    config.validate()?;
    let ingest_fingerprint = embedder.ingest_fingerprint(&config.chunk);
    let mut index = load_or_new(
        index_path,
        &config.model,
        embedder.dim(),
        ingest_fingerprint,
        reindex,
    )?;
    let mut stats = Stats::default();

    // Content-addressed embedding cache (keyed by embedding_key). Reused across
    // reindexes and projects so an unchanged chunk is not re-embedded. Best-effort
    // and disableable with `RAG_NO_EMB_CACHE=1`.
    let use_cache = std::env::var_os("RAG_NO_EMB_CACHE").is_none();
    let mut cache = if use_cache {
        EmbeddingCache::load(&config.model, embedder.dim())
    } else {
        EmbeddingCache::empty(embedder.dim())
    };

    let sources: Vec<PathBuf> = config.sources.iter().map(|s| root.join(s)).collect();
    let files = crate::walk::discover(&sources);
    let total_files = files.len();
    info!(files = total_files, "source scan complete");
    let discovered: HashSet<PathBuf> = files.iter().cloned().collect();

    // Prune files that vanished from disk / sources.
    let stale: Vec<PathBuf> = index
        .file_hashes
        .keys()
        .filter(|rel| !discovered.contains(&root.join(rel)))
        .cloned()
        .collect();
    for rel in stale {
        index.remove_file(&rel);
        stats.files_pruned += 1;
    }

    let mut progress = Progress::new(total_files);
    for (processed, path) in files.into_iter().enumerate() {
        if let Some(percent) = progress.update(processed) {
            info!(
                percent,
                processed,
                total = total_files,
                file = %path.display(),
                "indexing source files"
            );
        }
        let Some(lang) = Lang::from_path(&path) else {
            continue;
        };
        let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(error) => {
                warn!(file = %path.display(), %error, "skipping unreadable or non-UTF-8 file");
                index.remove_file(&rel);
                stats.files_failed += 1;
                continue;
            }
        };
        let hash = xxhash_rust::xxh3::xxh3_64(content.as_bytes());

        if !reindex && index.file_hashes.get(&rel) == Some(&hash) {
            stats.files_skipped += 1;
            continue;
        }

        index.remove_file(&rel);
        let chunks = chunk::chunk_file(&content, lang, &config.chunk)
            .with_context(|| format!("chunking {}", path.display()))?;
        if chunks.is_empty() {
            index.file_hashes.insert(rel, hash);
            continue;
        }

        // Second pass over the file: extract structural metadata per chunk and
        // build the ranking text that is actually embedded (raw text is stored
        // for the snippet). Metadata degrades to absent fields, never fails.
        let rel_str = rel.to_string_lossy();
        // File-level provenance (source URL + fetch date) from a leading
        // frontmatter block, if any; inherited by every chunk of this file.
        let provenance = metadata::frontmatter_provenance(&content);
        let analysis = metadata::FileAnalysis::new(&content, lang);
        let metas: Vec<metadata::ChunkMeta> = chunks
            .iter()
            .map(|c| analysis.meta_for(c.start_byte, c.end_byte))
            .collect();
        let ranking_texts: Vec<String> = chunks
            .iter()
            .zip(&metas)
            .map(|(c, m)| metadata::ranking_text(&rel_str, lang, m, &c.text))
            .collect();
        let keys: Vec<u64> = ranking_texts
            .iter()
            .map(|rt| metadata::embedding_key(ingest_fingerprint, rt))
            .collect();
        let (vectors, hits, misses) =
            embcache::resolve_with_cache(&mut cache, &keys, &ranking_texts, |miss| {
                embedder.embed_documents(miss)
            })?;
        stats.cache_hits += hits;
        stats.cache_misses += misses;
        for ((chunk, meta), (ranking_text, vector)) in chunks
            .into_iter()
            .zip(metas)
            .zip(ranking_texts.iter().zip(vectors))
        {
            index.records.push(Record {
                file: rel.clone(),
                lang: lang.name().to_string(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                chunk_id: metadata::chunk_id(
                    &rel_str,
                    lang,
                    chunk.start_byte,
                    chunk.end_byte,
                    &chunk.text,
                ),
                symbol: meta.symbol,
                kind: meta.kind,
                parent: meta.parent,
                signature: meta.signature,
                headings: meta.headings,
                source: provenance.source.clone(),
                corpus_source: provenance.corpus_source.clone(),
                fetched: provenance.fetched.clone(),
                embedding_key: metadata::embedding_key(ingest_fingerprint, ranking_text),
                text: chunk.text,
                vector,
            });
            stats.chunks += 1;
        }
        index.file_hashes.insert(rel, hash);
        stats.files_indexed += 1;
    }

    if let Some(percent) = progress.update(total_files) {
        info!(
            percent,
            processed = total_files,
            total = total_files,
            "source processing complete"
        );
    }
    if use_cache {
        cache.save();
    }
    info!(records = index.len(), "saving index");
    index.save(index_path)?;
    info!(path = %index_path.display(), "index saved");
    Ok((index, stats))
}

fn load_or_new(
    index_path: &Path,
    model: &str,
    dim: usize,
    ingest_fingerprint: u64,
    reindex: bool,
) -> Result<Index> {
    if reindex || !index_path.exists() {
        return Ok(Index::new(model, dim, ingest_fingerprint));
    }
    let existing = Index::load(index_path).with_context(|| {
        format!(
            "cannot reuse {}; run `rag-bone index --reindex` to rebuild it",
            index_path.display()
        )
    })?;
    if existing.model == model
        && existing.dim == dim
        && existing.ingest_fingerprint == ingest_fingerprint
    {
        Ok(existing)
    } else {
        info!(
            old_model = %existing.model,
            new_model = model,
            "ingestion settings changed; rebuilding the index"
        );
        Ok(Index::new(model, dim, ingest_fingerprint))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_or_new_rejects_model_mismatch() {
        let dir = std::env::temp_dir().join(format!("ragbone-idx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.bin");
        Index::new("old-model", 384, 1).save(&path).unwrap();
        let idx = load_or_new(&path, "new-model", 768, 2, false).unwrap();
        assert!(idx.is_empty());
        assert_eq!(idx.model, "new-model");
    }

    #[test]
    fn load_or_new_rejects_corrupt_index() {
        let dir = std::env::temp_dir().join(format!("ragbone-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.bin");
        std::fs::write(&path, b"not an index").unwrap();
        assert!(load_or_new(&path, "model", 384, 1, false).is_err());
    }
}
