//! In-memory BM25 lexical channel, derived at load time from the index records.
//!
//! Dense embeddings are weak on the queries a code agent actually issues —
//! symbols, paths, error strings, feature names, short keywords. This channel
//! complements them with exact term matching. It is deliberately *not* a
//! persisted secondary index: it is rebuilt from `index.records` each process,
//! so it is always in sync with the vectors (no schema bump, no coherence
//! failure mode, no silent drift). At ~1.3k chunks the rebuild is sub-millisecond;
//! revisit a persisted engine (Tantivy) only at the Fase 7 scale thresholds.
//!
//! Scoring is BM25F-lite: per-record fields (path, structural header, body) are
//! tokenized separately, their term frequencies weighted by a field boost, then
//! combined into one weighted bag with a single length normalization. A
//! code-aware tokenizer keeps each exact identifier *and* its snake_case /
//! camelCase pieces, so `search_document` matches both the whole token and
//! `document` without stemming.

use crate::store::Record;
use anyhow::Result;
use std::collections::HashMap;

/// BM25 term-frequency saturation.
const K1: f32 = 1.2;
/// BM25 length-normalization strength.
const B: f32 = 0.75;

/// Per-field boost applied to term frequencies before BM25 (BM25F). All 1.0 by
/// default, which makes the channel a plain BM25 over the concatenated fields —
/// the least-assumption baseline for the Fase 3 ablation. Field-weight tuning is
/// a separate, un-swept axis; raise these only with a measured benefit.
const PATH_BOOST: f32 = 1.0;
const STRUCTURAL_BOOST: f32 = 1.0;
const BODY_BOOST: f32 = 1.0;

/// One term's posting: the record index and its boosted term frequency.
type Posting = (usize, f32);

/// Derived BM25 index over the loaded records. `doc_len[i]` and the postings use
/// the same record indices as `Index::records`, so a hit maps straight back.
pub struct LexicalIndex {
    postings: HashMap<String, Vec<Posting>>,
    doc_freq: HashMap<String, usize>,
    doc_len: Vec<f32>,
    avgdl: f32,
    n_docs: usize,
}

impl LexicalIndex {
    /// Build the channel from every record. `include_structural` toggles whether
    /// the deterministic structural header (symbol/kind/parent/signature/headings)
    /// joins the lexical document — the "ranking_text vs raw" ablation axis.
    /// Fails loudly if no record yields a single term, rather than letting a
    /// hybrid/bm25 query silently degrade.
    pub fn build(records: &[Record], include_structural: bool) -> Result<Self> {
        let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut doc_len = Vec::with_capacity(records.len());
        let mut total_len = 0f32;

        for record in records {
            let mut weighted: HashMap<String, f32> = HashMap::new();
            add_field(&mut weighted, &record.file.to_string_lossy(), PATH_BOOST);
            if include_structural {
                for text in structural_fields(record) {
                    add_field(&mut weighted, &text, STRUCTURAL_BOOST);
                }
            }
            add_field(&mut weighted, &record.text, BODY_BOOST);

            let len: f32 = weighted.values().sum();
            doc_len.push(len);
            total_len += len;
            let idx = doc_len.len() - 1;
            for (term, wtf) in weighted {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
                postings.entry(term).or_default().push((idx, wtf));
            }
        }

        let n_docs = records.len();
        anyhow::ensure!(
            !postings.is_empty(),
            "lexical channel is empty (no indexable terms) — rebuild the index with `rag-bone index --reindex`"
        );
        let avgdl = if n_docs > 0 {
            total_len / n_docs as f32
        } else {
            0.0
        };
        Ok(Self {
            postings,
            doc_freq,
            doc_len,
            avgdl,
            n_docs,
        })
    }

    /// Top-`n` records by BM25, considering only records for which `allowed`
    /// returns true (the same lang/path filter the dense channel applies). Returns
    /// `(record_index, score)` best-first.
    pub fn search<F>(&self, query: &str, n: usize, allowed: F) -> Vec<(usize, f32)>
    where
        F: Fn(usize) -> bool,
    {
        let mut scores: HashMap<usize, f32> = HashMap::new();
        // A query term repeated adds nothing to BM25 here, so dedup it.
        let mut seen = std::collections::HashSet::new();
        for term in tokenize(query) {
            if !seen.insert(term.clone()) {
                continue;
            }
            let (Some(postings), Some(&df)) = (self.postings.get(&term), self.doc_freq.get(&term))
            else {
                continue;
            };
            let idf = idf(self.n_docs, df);
            for &(doc, wtf) in postings {
                if !allowed(doc) {
                    continue;
                }
                let norm = wtf + K1 * (1.0 - B + B * self.doc_len[doc] / self.avgdl);
                *scores.entry(doc).or_insert(0.0) += idf * (wtf * (K1 + 1.0)) / norm;
            }
        }
        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        // Stable, deterministic order: score desc, then record index asc.
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(n);
        ranked
    }
}

/// Structural header fields of a record, for the lexical symbol channel.
fn structural_fields(record: &Record) -> Vec<String> {
    let mut fields = Vec::new();
    fields.extend(record.symbol.clone());
    fields.extend(record.kind.clone());
    fields.extend(record.parent.clone());
    fields.extend(record.signature.clone());
    if !record.headings.is_empty() {
        fields.push(record.headings.join(" "));
    }
    fields
}

/// Tokenize a field and fold its (boosted) term frequencies into `weighted`.
fn add_field(weighted: &mut HashMap<String, f32>, text: &str, boost: f32) {
    for term in tokenize(text) {
        *weighted.entry(term).or_insert(0.0) += boost;
    }
}

/// BM25 inverse document frequency (with the standard +0.5 smoothing).
fn idf(n_docs: usize, df: usize) -> f32 {
    let n = n_docs as f32;
    let df = df as f32;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// Code-aware tokenizer. Each maximal identifier (`[alphanumeric or _]`) is kept
/// whole *and* split on `_` and camelCase into sub-tokens; everything is
/// lowercased. Separators (`::`, `.`, `/`, whitespace, punctuation) split
/// identifiers apart. The exact identifier is never lost, so `Index::save`,
/// `search_document` and `HTTPServer` each keep their whole form and their parts.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for ident in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if ident.is_empty() {
            continue;
        }
        let whole = ident.to_lowercase();
        for part in ident.split('_').filter(|p| !p.is_empty()) {
            for sub in camel_split(part) {
                let sub = sub.to_lowercase();
                if sub != whole {
                    out.push(sub);
                }
            }
        }
        out.push(whole);
    }
    out
}

/// Split an identifier part on camelCase boundaries: `lower→upper` (`fooBar`) and
/// the tail of an acronym run before a lowercase (`HTTPServer` → `HTTP`, `Server`).
/// Digits stay attached (`utf8`, `bm25` are not split), which keeps versioned
/// identifiers matchable.
fn camel_split(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 2 {
        return vec![word.to_string()];
    }
    let mut parts = Vec::new();
    let mut start = 0;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let lower_to_upper = prev.is_lowercase() && cur.is_uppercase();
        let acronym_tail = prev.is_uppercase()
            && cur.is_uppercase()
            && chars.get(i + 1).is_some_and(|n| n.is_lowercase());
        if lower_to_upper || acronym_tail {
            parts.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    parts.push(chars[start..].iter().collect());
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn rec(file: &str, symbol: Option<&str>, text: &str) -> Record {
        Record {
            file: PathBuf::from(file),
            lang: "rust".into(),
            start_line: 1,
            end_line: 2,
            chunk_id: format!("1-{file}-{text}"),
            symbol: symbol.map(String::from),
            kind: symbol.map(|_| "function".to_string()),
            parent: None,
            signature: symbol.map(|s| format!("fn {s}()")),
            headings: Vec::new(),
            source: None,
            corpus_source: None,
            fetched: None,
            embedding_key: 0,
            text: text.into(),
            vector: vec![1.0, 0.0],
        }
    }

    #[test]
    fn tokenize_keeps_whole_identifier_and_parts() {
        let t = tokenize("search_document");
        assert!(t.contains(&"search_document".to_string()));
        assert!(t.contains(&"search".to_string()));
        assert!(t.contains(&"document".to_string()));
    }

    #[test]
    fn tokenize_splits_camel_case_and_acronyms() {
        let t = tokenize("HTTPServer fooBar");
        assert!(t.contains(&"http".to_string()));
        assert!(t.contains(&"server".to_string()));
        assert!(t.contains(&"foo".to_string()));
        assert!(t.contains(&"bar".to_string()));
        assert!(t.contains(&"httpserver".to_string()));
    }

    #[test]
    fn tokenize_splits_path_and_scope_separators() {
        let t = tokenize("src/store.rs Index::save");
        for expected in ["src", "store", "rs", "index", "save"] {
            assert!(t.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn tokenize_keeps_versioned_identifier_whole() {
        // Digits stay attached: no `bm`, `25` split.
        let t = tokenize("bm25 utf8");
        assert!(t.contains(&"bm25".to_string()));
        assert!(t.contains(&"utf8".to_string()));
        assert!(!t.contains(&"bm".to_string()));
    }

    #[test]
    fn single_word_identifier_is_not_duplicated() {
        // `save` == its only sub-token, so it appears exactly once (tf stays 1).
        let t = tokenize("save");
        assert_eq!(t.iter().filter(|w| *w == "save").count(), 1);
    }

    #[test]
    fn bm25_ranks_exact_symbol_match_first() {
        let records = vec![
            rec("src/edit.rs", Some("edit_file"), "apply an edit to a file"),
            rec(
                "src/write.rs",
                Some("write_file"),
                "write a new file to disk",
            ),
            rec("src/read.rs", Some("read_file"), "read a file from disk"),
        ];
        let lex = LexicalIndex::build(&records, true).unwrap();
        let hits = lex.search("write_file", 3, |_| true);
        assert_eq!(hits[0].0, 1, "write.rs should rank first for write_file");
    }

    #[test]
    fn search_respects_the_filter_predicate() {
        let records = vec![
            rec("src/write.rs", Some("write_file"), "write a new file"),
            rec("docs/write.md", None, "write a new file"),
        ];
        let lex = LexicalIndex::build(&records, true).unwrap();
        // Only allow the second record: the first must not appear.
        let hits = lex.search("write file", 3, |idx| idx == 1);
        assert!(hits.iter().all(|&(idx, _)| idx == 1));
    }

    #[test]
    fn structural_toggle_changes_symbol_visibility() {
        // The symbol lives only in the structural header, not the body.
        let records = vec![
            rec("a.rs", Some("quota_window"), "unrelated body text"),
            rec("b.rs", None, "some other unrelated body"),
        ];
        let with = LexicalIndex::build(&records, true).unwrap();
        assert!(!with.search("quota_window", 3, |_| true).is_empty());
        let without = LexicalIndex::build(&records, false).unwrap();
        assert!(without.search("quota_window", 3, |_| true).is_empty());
    }

    #[test]
    fn empty_corpus_is_rejected() {
        assert!(LexicalIndex::build(&[], true).is_err());
    }
}
