use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const INDEX_SCHEMA_VERSION: u32 = 4;
const INDEX_FILE_MAGIC: &[u8; 8] = b"RAGBONE\0";
/// Fixed header: magic (8) + schema version u32 LE (4) + scratch bound u64 LE (8).
/// The schema version lives in the header, not just the postcard body, so a
/// schema change is rejected with a `--reindex` hint *before* the record layout
/// is decoded (a layout change otherwise fails mid-decode with a cryptic error).
const INDEX_HEADER_LEN: u64 = 20;
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One stored chunk: its provenance, structural metadata, raw text, and
/// normalized embedding. `text` is the raw snippet returned to the caller; the
/// embedded/ranking text is reconstructed from the metadata, not stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub file: PathBuf,
    pub lang: String,
    pub start_line: usize,
    pub end_line: usize,
    /// Public, versioned, deterministic chunk identifier (for `get`/dedup).
    pub chunk_id: String,
    /// Enclosing definition name, kind, container and one-line signature (code),
    /// or empty for prose. Optional fields degrade to `None`.
    pub symbol: Option<String>,
    pub kind: Option<String>,
    pub parent: Option<String>,
    pub signature: Option<String>,
    /// Markdown heading breadcrumb (prose only).
    pub headings: Vec<String>,
    /// File-level provenance from a Markdown frontmatter block: the origin URL
    /// and fetch timestamp of a web-sourced document (search2md). `None` for
    /// local files without frontmatter. Surfaced in results so an agent can cite
    /// where a chunk came from and how old it is.
    pub source: Option<String>,
    pub corpus_source: Option<String>,
    pub fetched: Option<String>,
    /// Internal cache key: hash of the ranking text bound to the ingest
    /// fingerprint, excluding the line span. Consumed by the embedding cache.
    pub embedding_key: u64,
    /// Raw chunk text, returned as the result snippet.
    pub text: String,
    /// L2-normalized embedding (cosine == dot product).
    pub vector: Vec<f32>,
}

/// The whole persisted index: metadata, per-file content hashes for incremental
/// reindexing, and the flat record list searched by brute-force cosine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub schema_version: u32,
    pub ingest_fingerprint: u64,
    pub model: String,
    pub dim: usize,
    pub file_hashes: HashMap<PathBuf, u64>,
    pub records: Vec<Record>,
}

/// One web-sourced document set in the index: origin URL, fetch timestamp, and
/// how many files/chunks it contributed. The agent's routing menu (see `catalog`).
#[derive(Debug, Clone, Serialize)]
pub struct CatalogEntry {
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corpus_source: Option<String>,
    pub fetched: Option<String>,
    pub files: usize,
    pub chunks: usize,
}

/// A search hit: index into `records` and its cosine score.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub idx: usize,
    pub score: f32,
}

impl Index {
    pub fn new(model: impl Into<String>, dim: usize, ingest_fingerprint: u64) -> Self {
        Self {
            schema_version: INDEX_SCHEMA_VERSION,
            ingest_fingerprint,
            model: model.into(),
            dim,
            file_hashes: HashMap::new(),
            records: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Number of distinct indexed files.
    pub fn file_count(&self) -> usize {
        self.file_hashes.len()
    }

    /// Stream an index from disk, then validate its schema and vector dimensions.
    pub fn load(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let mut magic = [0; 8];
        reader
            .read_exact(&mut magic)
            .with_context(|| format!("reading header from {}", path.display()))?;
        anyhow::ensure!(
            &magic == INDEX_FILE_MAGIC,
            "{} is not a rag-bone index",
            path.display()
        );
        let mut schema_bytes = [0; 4];
        reader
            .read_exact(&mut schema_bytes)
            .with_context(|| format!("reading header from {}", path.display()))?;
        let schema_version = u32::from_le_bytes(schema_bytes);
        anyhow::ensure!(
            schema_version == INDEX_SCHEMA_VERSION,
            "unsupported index schema {} in {}; expected {} — rebuild with `rag-bone index --reindex`",
            schema_version,
            path.display(),
            INDEX_SCHEMA_VERSION
        );
        let mut scratch_bytes = [0; 8];
        reader
            .read_exact(&mut scratch_bytes)
            .with_context(|| format!("reading header from {}", path.display()))?;
        let scratch_len = usize::try_from(u64::from_le_bytes(scratch_bytes))
            .context("index scratch size does not fit this platform")?;
        let file_len = reader
            .get_ref()
            .metadata()
            .with_context(|| format!("reading metadata for {}", path.display()))?
            .len();
        anyhow::ensure!(
            scratch_len > 0 && (scratch_len as u64) <= file_len.saturating_sub(INDEX_HEADER_LEN),
            "invalid scratch size in {}",
            path.display()
        );
        let mut scratch = vec![0; scratch_len];
        let (index, _): (Self, _) = postcard::from_io((&mut reader, &mut scratch))
            .with_context(|| format!("decoding {}", path.display()))?;
        anyhow::ensure!(
            reader.fill_buf()?.is_empty(),
            "{} has trailing data",
            path.display()
        );
        index
            .validate()
            .with_context(|| format!("validating {}", path.display()))?;
        Ok(index)
    }

    /// Persist through a same-directory temporary file. Unix uses an atomic
    /// replacement; Windows uses the best available std-only replacement path.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate().context("validating index before save")?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
        let temp_path = temporary_path(path, parent);

        let write_result = (|| -> Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
                .with_context(|| format!("creating {}", temp_path.display()))?;
            {
                let mut writer = BufWriter::new(&mut file);
                writer
                    .write_all(INDEX_FILE_MAGIC)
                    .context("writing index header")?;
                writer
                    .write_all(&INDEX_SCHEMA_VERSION.to_le_bytes())
                    .context("writing index header")?;
                writer
                    .write_all(&(self.scratch_len() as u64).to_le_bytes())
                    .context("writing index header")?;
                postcard::to_io(self, &mut writer).context("encoding index")?;
                writer
                    .flush()
                    .with_context(|| format!("flushing {}", temp_path.display()))?;
            }
            file.sync_all()
                .with_context(|| format!("syncing {}", temp_path.display()))?;
            replace_index_file(&temp_path, path)?;
            sync_parent(parent)?;
            Ok(())
        })();

        if write_result.is_err() {
            let _ = std::fs::remove_file(&temp_path);
        }
        write_result
    }

    /// Find a record by its public `chunk_id` (for the `get` command). Ids are
    /// unique per index (enforced at save), so the first match is the only one.
    pub fn get_by_chunk_id(&self, chunk_id: &str) -> Option<&Record> {
        self.records.iter().find(|r| r.chunk_id == chunk_id)
    }

    /// Structural outline of a file: its chunks (whose path contains `file_query`)
    /// in source order. Powers `outline` — a models-free view of a file's symbols.
    pub fn outline(&self, file_query: &str) -> Vec<&Record> {
        let mut hits: Vec<&Record> = self
            .records
            .iter()
            .filter(|r| r.file.to_string_lossy().contains(file_query))
            .collect();
        hits.sort_by(|a, b| a.file.cmp(&b.file).then(a.start_line.cmp(&b.start_line)));
        hits
    }

    /// Records defining a symbol named `name`, exact (case-insensitive) matches
    /// first, then symbols merely containing `name`. Powers `find` — a models-free
    /// jump-to-definition that does not depend on semantic ranking.
    pub fn find_symbol(&self, name: &str) -> Vec<&Record> {
        let needle = name.to_lowercase();
        let mut hits: Vec<(u8, &Record)> = self
            .records
            .iter()
            .filter_map(|r| {
                let symbol = r.symbol.as_ref()?.to_lowercase();
                if symbol == needle {
                    Some((0, r))
                } else if symbol.contains(&needle) {
                    Some((1, r))
                } else {
                    None
                }
            })
            .collect();
        hits.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.file.cmp(&b.1.file))
                .then(a.1.start_line.cmp(&b.1.start_line))
        });
        hits.into_iter().map(|(_, r)| r).collect()
    }

    /// Group indexed records by their `source` provenance into one entry per
    /// web-fetched document set (origin URL, fetch date, file/chunk counts).
    /// Local files without a `source` are omitted. Powers `catalog` — the cheap,
    /// models-free "what external docs exist and how old" menu an agent reads
    /// before scoping a query with `--source`. Sorted by source URL.
    pub fn catalog(&self) -> Vec<CatalogEntry> {
        use std::collections::{BTreeMap, HashSet};
        type SourceAggregate<'a> = (Option<&'a str>, Option<&'a str>, HashSet<&'a Path>, usize);
        let mut by_source: BTreeMap<&str, SourceAggregate<'_>> = BTreeMap::new();
        for record in &self.records {
            let Some(source) = record.source.as_deref() else {
                continue;
            };
            let entry = by_source
                .entry(source)
                .or_insert_with(|| (None, None, HashSet::new(), 0));
            if entry.0.is_none() {
                entry.0 = record.corpus_source.as_deref();
            }
            if entry.1.is_none() {
                entry.1 = record.fetched.as_deref();
            }
            entry.2.insert(record.file.as_path());
            entry.3 += 1;
        }
        by_source
            .into_iter()
            .map(
                |(source, (corpus_source, fetched, files, chunks))| CatalogEntry {
                    source: source.to_string(),
                    corpus_source: corpus_source.map(str::to_string),
                    fetched: fetched.map(str::to_string),
                    files: files.len(),
                    chunks,
                },
            )
            .collect()
    }

    /// Drop every record belonging to `file` (used before reindexing it).
    pub fn remove_file(&mut self, file: &Path) {
        self.records.retain(|r| r.file != file);
        self.file_hashes.remove(file);
    }

    /// Top-`n` records after applying a metadata predicate. Filtering before the
    /// top-N cut prevents a selective language/path filter from losing candidates.
    /// Brute-force, parallelized across cores — exact, not approximate.
    pub fn search_filtered<F>(&self, query: &[f32], n: usize, predicate: F) -> Result<Vec<Hit>>
    where
        F: Fn(&Record) -> bool + Sync,
    {
        anyhow::ensure!(
            query.len() == self.dim,
            "query dimension {} does not match index dimension {}",
            query.len(),
            self.dim
        );
        let mut scored: Vec<Hit> = self
            .records
            .par_iter()
            .enumerate()
            .filter(|(_, record)| predicate(record))
            .map(|(idx, r)| Hit {
                idx,
                score: dot(&r.vector, query),
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(n);
        Ok(scored)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == INDEX_SCHEMA_VERSION,
            "unsupported index schema {}; expected {} — rebuild with `rag-bone index --reindex`",
            self.schema_version,
            INDEX_SCHEMA_VERSION
        );
        anyhow::ensure!(self.dim > 0, "index dimension must be greater than zero");
        let mut seen_ids = std::collections::HashSet::with_capacity(self.records.len());
        for record in &self.records {
            anyhow::ensure!(
                record.vector.len() == self.dim,
                "vector dimension {} for {} does not match index dimension {}",
                record.vector.len(),
                record.file.display(),
                self.dim
            );
            anyhow::ensure!(
                record.vector.iter().all(|value| value.is_finite()),
                "non-finite vector value in {}",
                record.file.display()
            );
            anyhow::ensure!(
                self.file_hashes.contains_key(&record.file),
                "record {} has no file hash",
                record.file.display()
            );
            // Build-time chunk-id collision check: ids are the public contract for
            // `get`/dedup, so a duplicate must fail loudly rather than silently
            // alias two chunks.
            anyhow::ensure!(
                seen_ids.insert(record.chunk_id.as_str()),
                "duplicate chunk id {} (collision on {})",
                record.chunk_id,
                record.file.display()
            );
        }
        Ok(())
    }

    /// `postcard::from_io` only needs temporary storage for the largest owned
    /// string/byte field, not for the full index. Persist that bound in the file
    /// header so loading stays streaming without a hard-coded chunk-size limit.
    fn scratch_len(&self) -> usize {
        self.file_hashes
            .keys()
            .map(|path| path.to_string_lossy().len())
            .chain(self.records.iter().flat_map(|record| {
                [
                    record.file.to_string_lossy().len(),
                    record.lang.len(),
                    record.text.len(),
                    record.chunk_id.len(),
                    record.symbol.as_ref().map_or(0, String::len),
                    record.parent.as_ref().map_or(0, String::len),
                    record.signature.as_ref().map_or(0, String::len),
                    record.headings.iter().map(String::len).max().unwrap_or(0),
                    record.source.as_ref().map_or(0, String::len),
                    record.corpus_source.as_ref().map_or(0, String::len),
                    record.fetched.as_ref().map_or(0, String::len),
                ]
            }))
            .chain([self.model.len(), std::mem::size_of::<f64>()])
            .max()
            .unwrap_or(1)
    }
}

fn temporary_path(path: &Path, parent: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map_or_else(|| "index".into(), |name| name.to_string_lossy());
    let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(".{name}.tmp-{}-{sequence}", std::process::id()))
}

#[cfg(not(windows))]
fn replace_index_file(temp_path: &Path, path: &Path) -> Result<()> {
    std::fs::rename(temp_path, path)
        .with_context(|| format!("replacing {} with {}", path.display(), temp_path.display()))
}

#[cfg(windows)]
fn replace_index_file(temp_path: &Path, path: &Path) -> Result<()> {
    match std::fs::rename(temp_path, path) {
        Ok(()) => Ok(()),
        Err(_) if path.exists() => {
            std::fs::remove_file(path)
                .with_context(|| format!("removing old index {}", path.display()))?;
            std::fs::rename(temp_path, path).with_context(|| {
                format!("replacing {} with {}", path.display(), temp_path.display())
            })
        }
        Err(error) => Err(error)
            .with_context(|| format!("moving {} to {}", temp_path.display(), path.display())),
    }
}

#[cfg(unix)]
fn sync_parent(parent: &Path) -> Result<()> {
    std::fs::File::open(parent)
        .with_context(|| format!("opening {} for sync", parent.display()))?
        .sync_all()
        .with_context(|| format!("syncing {}", parent.display()))
}

#[cfg(not(unix))]
fn sync_parent(_parent: &Path) -> Result<()> {
    Ok(())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(file: &str, v: Vec<f32>) -> Record {
        Record {
            file: PathBuf::from(file),
            lang: "rust".into(),
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
            vector: v,
        }
    }

    #[test]
    fn search_ranks_by_cosine() {
        let mut idx = Index::new("m", 2, 1);
        idx.records = vec![
            rec("a.rs", vec![1.0, 0.0]),
            rec("b.rs", vec![0.0, 1.0]),
            rec("c.rs", vec![std::f32::consts::FRAC_1_SQRT_2; 2]),
        ];
        idx.file_hashes.insert(PathBuf::from("a.rs"), 1);
        idx.file_hashes.insert(PathBuf::from("b.rs"), 2);
        idx.file_hashes.insert(PathBuf::from("c.rs"), 3);
        let hits = idx.search_filtered(&[1.0, 0.0], 2, |_| true).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].idx, 0);
        assert_eq!(hits[1].idx, 2);
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ragbone-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.bin");
        let mut idx = Index::new("nomic", 2, 7);
        idx.records = vec![rec("a.rs", vec![1.0, 0.0])];
        idx.file_hashes.insert(PathBuf::from("a.rs"), 42);
        idx.save(&path).unwrap();
        let loaded = Index::load(&path).unwrap();
        assert_eq!(loaded.model, "nomic");
        assert_eq!(loaded.ingest_fingerprint, 7);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.file_hashes.get(Path::new("a.rs")), Some(&42));
    }

    #[test]
    fn save_replaces_existing_index_with_large_record() {
        let dir = std::env::temp_dir().join(format!("ragbone-replace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.bin");
        let mut idx = Index::new("model", 2, 1);
        idx.records = vec![rec("large.rs", vec![1.0, 0.0])];
        idx.file_hashes.insert(PathBuf::from("large.rs"), 1);
        idx.save(&path).unwrap();

        idx.records[0].text = "x".repeat(100_000);
        idx.file_hashes.insert(PathBuf::from("large.rs"), 2);
        idx.save(&path).unwrap();

        let loaded = Index::load(&path).unwrap();
        assert_eq!(loaded.records[0].text.len(), 100_000);
        assert_eq!(loaded.file_hashes.get(Path::new("large.rs")), Some(&2));
        assert!(std::fs::read_dir(&dir).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        }));
    }

    #[test]
    fn load_rejects_old_schema_with_reindex_hint() {
        let dir = std::env::temp_dir().join(format!("ragbone-schema-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.bin");
        let mut idx = Index::new("m", 2, 1);
        idx.records = vec![rec("a.rs", vec![1.0, 0.0])];
        idx.file_hashes.insert(PathBuf::from("a.rs"), 1);
        idx.save(&path).unwrap();
        // Corrupt the schema version in the fixed header (bytes 8..12).
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&(INDEX_SCHEMA_VERSION + 1).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = Index::load(&path).unwrap_err().to_string();
        assert!(err.contains("--reindex"), "unexpected error: {err}");
    }

    #[test]
    fn get_by_chunk_id_finds_and_misses() {
        let mut idx = Index::new("m", 2, 1);
        idx.records = vec![rec("a.rs", vec![1.0, 0.0]), rec("b.rs", vec![0.0, 1.0])];
        assert_eq!(
            idx.get_by_chunk_id("1-b.rs").map(|r| r.file.clone()),
            Some(PathBuf::from("b.rs"))
        );
        assert!(idx.get_by_chunk_id("nope").is_none());
    }

    #[test]
    fn catalog_groups_web_sources_and_omits_local() {
        let mut idx = Index::new("m", 2, 1);
        let mut a = rec("vendor/serde.md", vec![1.0, 0.0]);
        a.source = Some("https://docs.rs/serde".into());
        a.fetched = Some("2026-07-19T10:00:00Z".into());
        let mut a2 = rec("vendor/serde.md", vec![0.0, 1.0]);
        a2.chunk_id = "2-serde".into();
        a2.source = Some("https://docs.rs/serde".into());
        let local = rec("src/main.rs", vec![1.0, 0.0]);
        idx.records = vec![a, a2, local];
        let cat = idx.catalog();
        assert_eq!(cat.len(), 1, "local file omitted, web source grouped");
        assert_eq!(cat[0].source, "https://docs.rs/serde");
        assert_eq!(cat[0].chunks, 2);
        assert_eq!(cat[0].files, 1);
        assert_eq!(cat[0].fetched.as_deref(), Some("2026-07-19T10:00:00Z"));
    }

    fn rec_sym(file: &str, symbol: &str, start: usize) -> Record {
        let mut r = rec(file, vec![1.0, 0.0]);
        r.symbol = Some(symbol.into());
        r.kind = Some("function".into());
        r.start_line = start;
        r.chunk_id = format!("1-{file}-{start}");
        r
    }

    #[test]
    fn outline_returns_file_chunks_in_source_order() {
        let mut idx = Index::new("m", 2, 1);
        idx.records = vec![
            rec_sym("src/a.rs", "second", 50),
            rec_sym("src/a.rs", "first", 10),
            rec_sym("src/b.rs", "other", 1),
        ];
        let out = idx.outline("a.rs");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].symbol.as_deref(), Some("first")); // start_line 10 before 50
        assert_eq!(out[1].symbol.as_deref(), Some("second"));
    }

    #[test]
    fn find_symbol_ranks_exact_before_substring() {
        let mut idx = Index::new("m", 2, 1);
        idx.records = vec![
            rec_sym("src/a.rs", "save_all", 5), // substring
            rec_sym("src/b.rs", "save", 5),     // exact
            rec_sym("src/c.rs", "load", 5),     // no match
        ];
        let hits = idx.find_symbol("save");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].symbol.as_deref(), Some("save")); // exact first
        assert_eq!(hits[1].symbol.as_deref(), Some("save_all"));
        assert!(idx.find_symbol("missing").is_empty());
    }

    #[test]
    fn remove_file_drops_records() {
        let mut idx = Index::new("m", 2, 1);
        idx.records = vec![rec("a.rs", vec![1.0, 0.0]), rec("b.rs", vec![0.0, 1.0])];
        idx.file_hashes.insert(PathBuf::from("a.rs"), 1);
        idx.remove_file(Path::new("a.rs"));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.records[0].file, PathBuf::from("b.rs"));
    }

    #[test]
    fn search_filters_before_top_n() {
        let mut idx = Index::new("m", 2, 1);
        idx.records = vec![
            rec("docs/a.md", vec![1.0, 0.0]),
            rec("src/a.rs", vec![0.9, 0.1]),
        ];
        idx.records[0].lang = "md".into();
        idx.file_hashes.insert(PathBuf::from("docs/a.md"), 1);
        idx.file_hashes.insert(PathBuf::from("src/a.rs"), 2);

        let hits = idx
            .search_filtered(&[1.0, 0.0], 1, |record| record.lang == "rust")
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(idx.records[hits[0].idx].file, PathBuf::from("src/a.rs"));
    }

    #[test]
    fn rejects_wrong_query_dimension() {
        let idx = Index::new("m", 2, 1);
        assert!(idx.search_filtered(&[1.0], 1, |_| true).is_err());
    }
}
