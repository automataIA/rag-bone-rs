use crate::embed::Embedder;
use crate::progress::Progress;
use crate::rerank::Reranker;
use crate::search::{self, SearchParams, SearchResult};
use crate::store::Index;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// One golden-set row. Format: `query <TAB> a|b <TAB> category <TAB> lines`.
/// The first expected path is the primary answer (graded relevance 2), any `|`
/// alternates are secondary (grade 1). Columns 3 (`category`) and 4
/// (`expected_lines`, a `start-end` range in the primary file) are optional.
struct Case {
    query: String,
    expected: Vec<String>,
    category: String,
    /// Optional expected line range in the answer file, for the span-hit metric.
    expected_lines: Option<(usize, usize)>,
}

/// Category used when a golden-set row omits the third column.
const DEFAULT_CATEGORY: &str = "uncategorized";
/// Recall cutoff reported alongside `recall_at_k`. The eval retrieves at least
/// this many results per query so both cutoffs come from a single search.
const RECALL_WIDE_K: usize = 10;
/// nDCG cutoff.
const NDCG_K: usize = 10;

/// Retrieval knobs for a sweep run — mirror the `Config` fields the eval loop
/// needs, so one struct travels instead of four positional args.
pub struct EvalOpts {
    pub k: usize,
    pub retrieve_n: usize,
    pub rerank: bool,
    pub min_score: f32,
    pub retrieval: crate::config::RetrievalMode,
    pub rrf_k: usize,
    /// Optional JSONL sink: one line per query with the returned top-k, for the
    /// downstream LLM-as-judge relevance pass.
    pub dump: Option<std::path::PathBuf>,
}

/// Aggregate metrics over the whole golden set, plus a per-category breakdown.
pub struct EvalReport {
    pub n: usize,
    pub k: usize,
    pub recall_at_k: f32,
    pub recall_at_10: f32,
    pub mrr: f32,
    pub ndcg_at_10: f32,
    /// Span-hit rate and the number of queries carrying a range expectation, or
    /// `None` when no query in the set provides one.
    pub span: Option<(f32, usize)>,
    pub mean_latency_ms: f32,
    /// One entry per query category, sorted by category name. Small strata are
    /// noisy — read `n` before trusting a stratum's recall.
    pub strata: Vec<StratumReport>,
}

/// Metrics for one query stratum.
pub struct StratumReport {
    pub category: String,
    pub n: usize,
    pub recall_at_k: f32,
    pub recall_at_10: f32,
    pub mrr: f32,
    pub ndcg_at_10: f32,
    pub span: Option<(f32, usize)>,
}

/// One query's scored outcome, folded into a `Tally`.
struct Outcome {
    /// 0-indexed rank of the first path-matching result, or `None` on a miss.
    rank: Option<usize>,
    /// nDCG@10 for this query (path-graded relevance).
    ndcg: f32,
    /// `Some(true/false)` when the query carries a line-range expectation:
    /// whether a top-k result matched the path *and* overlapped the range.
    span_hit: Option<bool>,
}

/// Running counters for one stratum (or the whole set), folded into rates at the end.
#[derive(Default)]
struct Tally {
    n: usize,
    hits_k: usize,
    hits_10: usize,
    rr_sum: f32,
    ndcg_sum: f32,
    /// Span metric is averaged only over queries that provide a range.
    span_n: usize,
    span_hits: usize,
}

impl Tally {
    fn record(&mut self, outcome: &Outcome, k: usize) {
        self.n += 1;
        if let Some(pos) = outcome.rank {
            if pos < k {
                self.hits_k += 1;
            }
            if pos < RECALL_WIDE_K {
                self.hits_10 += 1;
            }
            self.rr_sum += 1.0 / (pos as f32 + 1.0);
        }
        self.ndcg_sum += outcome.ndcg;
        if let Some(hit) = outcome.span_hit {
            self.span_n += 1;
            self.span_hits += usize::from(hit);
        }
    }

    fn recall_k(&self) -> f32 {
        self.hits_k as f32 / self.n as f32
    }
    fn recall_10(&self) -> f32 {
        self.hits_10 as f32 / self.n as f32
    }
    fn mrr(&self) -> f32 {
        self.rr_sum / self.n as f32
    }
    fn ndcg(&self) -> f32 {
        self.ndcg_sum / self.n as f32
    }
    /// Span-hit rate over queries with a range expectation, and their count.
    fn span_hit_rate(&self) -> Option<(f32, usize)> {
        (self.span_n > 0).then(|| (self.span_hits as f32 / self.span_n as f32, self.span_n))
    }
}

/// Load the models once and evaluate every query in-process — the whole point
/// of this subcommand is to pay the ~15s model cold-start a single time instead
/// of once per query like the external `score.sh` fallback does.
pub fn run(
    index: &Index,
    embedder: &mut Embedder,
    mut reranker: Option<&mut Reranker>,
    lexical: Option<&crate::lexical::LexicalIndex>,
    queries_path: &Path,
    opts: &EvalOpts,
) -> Result<EvalReport> {
    let cases = load_cases(queries_path)?;
    anyhow::ensure!(
        !cases.is_empty(),
        "no queries in {}",
        queries_path.display()
    );

    let mut overall = Tally::default();
    let mut per_category: BTreeMap<String, Tally> = BTreeMap::new();
    let mut latency_sum = 0f32;
    // Retrieve enough results to score both recall cutoffs from one search.
    let wide_k = opts.k.max(RECALL_WIDE_K);

    let mut dump = opts
        .dump
        .as_ref()
        .map(|p| std::fs::File::create(p).with_context(|| format!("creating {}", p.display())))
        .transpose()?;

    let mut progress = Progress::new(cases.len());
    for (completed, case) in cases.iter().enumerate() {
        if let Some(percent) = progress.update(completed) {
            tracing::info!(
                percent,
                completed,
                total = cases.len(),
                "evaluating golden set"
            );
        }
        let params = SearchParams {
            query: case.query.clone(),
            top_k: wide_k,
            retrieve_n: opts.retrieve_n.max(wide_k),
            rerank: opts.rerank,
            min_score: opts.min_score,
            langs: None,
            path_prefix: None,
            source: None,
            retrieval: opts.retrieval,
            rrf_k: opts.rrf_k,
            // Eval scores raw ranking quality; diversification is a delivery-time
            // knob, kept off here so metrics reflect the ranker, not the cap.
            max_per_file: 0,
        };
        let start = Instant::now();
        let results = search::run(
            index,
            Some(&mut *embedder),
            reranker.as_deref_mut(),
            lexical,
            &params,
        )?;
        latency_sum += start.elapsed().as_secs_f32() * 1000.0;

        // 0-indexed rank of the first result whose path matches any expected
        // substring; feeds recall@k, recall@10 and reciprocal rank.
        let rank = results
            .iter()
            .position(|r| matches_expected(&r.file.to_string_lossy(), &case.expected));
        let outcome = Outcome {
            rank,
            ndcg: ndcg_at_10(&results, &case.expected),
            span_hit: case
                .expected_lines
                .map(|range| span_hit(&results, &case.expected, range, opts.k)),
        };
        overall.record(&outcome, opts.k);
        per_category
            .entry(case.category.clone())
            .or_default()
            .record(&outcome, opts.k);

        if let Some(file) = dump.as_mut() {
            // Preserve the original dump shape (top-k) for the downstream judge,
            // independent of the wider retrieval used for recall@10.
            write_dump_line(file, case, &results[..opts.k.min(results.len())])?;
        }

        // Per-query visibility for golden-set validation (RAG_EVAL_DEBUG=1):
        // show misses and the top path actually returned.
        if std::env::var_os("RAG_EVAL_DEBUG").is_some() {
            let mark = rank.map_or("MISS", |_| "hit ");
            let top = results
                .first()
                .map_or("-", |r| r.file.to_str().unwrap_or("-"));
            eprintln!("{mark} rank={rank:?}  top={top}  q={}", case.query);
        }
    }

    if let Some(percent) = progress.update(cases.len()) {
        tracing::info!(
            percent,
            completed = cases.len(),
            total = cases.len(),
            "golden-set evaluation complete"
        );
    }

    let strata = per_category
        .into_iter()
        .map(|(category, t)| StratumReport {
            category,
            n: t.n,
            recall_at_k: t.recall_k(),
            recall_at_10: t.recall_10(),
            mrr: t.mrr(),
            ndcg_at_10: t.ndcg(),
            span: t.span_hit_rate(),
        })
        .collect();

    Ok(EvalReport {
        n: overall.n,
        k: opts.k,
        recall_at_k: overall.recall_k(),
        recall_at_10: overall.recall_10(),
        mrr: overall.mrr(),
        ndcg_at_10: overall.ndcg(),
        span: overall.span_hit_rate(),
        mean_latency_ms: latency_sum / overall.n as f32,
        strata,
    })
}

/// nDCG@10 with path-graded relevance: the primary expected path scores gain 2,
/// any `|` alternates gain 1. Each relevant path is credited once (the first
/// retrieved chunk from it); further chunks of the same file add nothing.
fn ndcg_at_10(results: &[SearchResult], expected: &[String]) -> f32 {
    let mut credited = std::collections::HashSet::new();
    let mut dcg = 0.0f32;
    for (rank, r) in results.iter().take(NDCG_K).enumerate() {
        if let Some(idx) = best_match_index(&r.file.to_string_lossy(), expected)
            && credited.insert(idx)
        {
            let gain = if idx == 0 { 3.0 } else { 1.0 }; // 2^grade - 1
            dcg += gain / (rank as f32 + 2.0).log2();
        }
    }
    // Ideal DCG: primary (gain 3) first, then each alternate (gain 1).
    let mut ideal: Vec<f32> = expected
        .iter()
        .enumerate()
        .map(|(i, _)| if i == 0 { 3.0 } else { 1.0 })
        .collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f32 = ideal
        .iter()
        .enumerate()
        .map(|(i, g)| g / (i as f32 + 2.0).log2())
        .sum();
    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

/// Index into `expected` of the relevant path a result satisfies, preferring the
/// primary (index 0) when it matches. `None` if the result matches no expected path.
fn best_match_index(path: &str, expected: &[String]) -> Option<usize> {
    if expected.first().is_some_and(|e| path.contains(e.as_str())) {
        return Some(0);
    }
    expected.iter().position(|e| path.contains(e.as_str()))
}

/// Whether a top-`k` result matches an expected path and its line span overlaps
/// the expected range `[lo, hi]`.
fn span_hit(
    results: &[SearchResult],
    expected: &[String],
    (lo, hi): (usize, usize),
    k: usize,
) -> bool {
    results.iter().take(k).any(|r| {
        matches_expected(&r.file.to_string_lossy(), expected)
            && r.lines[0] <= hi
            && lo <= r.lines[1]
    })
}

/// One JSONL record: the query, its accepted paths, and the returned top-k —
/// the exact input the LLM judge scores for context relevance.
fn write_dump_line(file: &mut std::fs::File, case: &Case, results: &[SearchResult]) -> Result<()> {
    let record = serde_json::json!({
        "query": case.query,
        "expected": case.expected,
        "results": results,
    });
    writeln!(file, "{}", serde_json::to_string(&record)?).context("writing dump line")
}

fn matches_expected(path: &str, expected: &[String]) -> bool {
    expected.iter().any(|e| path.contains(e.as_str()))
}

/// Parse a `start-end` line range (1-based, inclusive). Returns `None` on any
/// malformed or empty field, so a blank fourth column simply disables span-hit.
fn parse_line_range(field: &str) -> Option<(usize, usize)> {
    let (lo, hi) = field.trim().split_once('-')?;
    let lo: usize = lo.trim().parse().ok()?;
    let hi: usize = hi.trim().parse().ok()?;
    (lo <= hi).then_some((lo, hi))
}

/// Parse the tsv: skip the header line and blanks, split
/// `query <TAB> expected <TAB> category <TAB> lines`, and split expected on `|`
/// into alternatives. Columns 3 (`category`) and 4 (`expected_lines`, a
/// `start-end` range) are optional (older files without them still parse).
fn load_cases(path: &Path) -> Result<Vec<Case>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading queries {}", path.display()))?;
    Ok(text
        .lines()
        .skip(1) // header
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let mut fields = line.splitn(4, '\t');
            let query = fields.next()?;
            let expected: Vec<String> = fields
                .next()?
                .split('|')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            let category = fields
                .next()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(DEFAULT_CATEGORY)
                .to_string();
            let expected_lines = fields.next().and_then(parse_line_range);
            (!expected.is_empty()).then(|| Case {
                query: query.trim().to_string(),
                expected,
                category,
                expected_lines,
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_any_alternative() {
        let exp = vec!["quota.rs".to_string(), "permissions.rs".to_string()];
        assert!(matches_expected("src/permissions.rs", &exp));
        assert!(!matches_expected("src/session.rs", &exp));
    }

    #[test]
    fn parse_skips_header_and_blanks() {
        let dir = std::env::temp_dir().join(format!("ragbone-eval-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q.tsv");
        std::fs::write(&path, "query\texpected\nfoo\ta.rs|b.rs\n\nbar\tc.md\n").unwrap();
        let cases = load_cases(&path).unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].expected, vec!["a.rs", "b.rs"]);
        assert_eq!(cases[1].query, "bar");
    }

    #[test]
    fn parse_reads_optional_category_column() {
        let dir = std::env::temp_dir().join(format!("ragbone-evalcat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q.tsv");
        // Row 1 has a category, row 2 omits it → falls back to the default.
        std::fs::write(
            &path,
            "query\texpected\tcategory\nfoo\ta.rs\tdocs\nbar\tb.rs\n",
        )
        .unwrap();
        let cases = load_cases(&path).unwrap();
        assert_eq!(cases[0].category, "docs");
        assert_eq!(cases[1].category, DEFAULT_CATEGORY);
    }

    fn outcome(rank: Option<usize>, span_hit: Option<bool>) -> Outcome {
        Outcome {
            rank,
            ndcg: 0.0,
            span_hit,
        }
    }

    #[test]
    fn tally_folds_ranks_into_rates() {
        let mut t = Tally::default();
        t.record(&outcome(Some(0), None), 3); // hit@k, hit@10, rr=1
        t.record(&outcome(Some(5), None), 3); // miss@k (5>=3), hit@10, rr=1/6
        t.record(&outcome(None, None), 3); // full miss
        assert!((t.recall_k() - 1.0 / 3.0).abs() < 1e-6);
        assert!((t.recall_10() - 2.0 / 3.0).abs() < 1e-6);
        assert!((t.mrr() - (1.0 + 1.0 / 6.0) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn tally_span_rate_only_counts_queries_with_a_range() {
        let mut t = Tally::default();
        t.record(&outcome(Some(0), Some(true)), 3);
        t.record(&outcome(Some(0), Some(false)), 3);
        t.record(&outcome(Some(0), None), 3); // no range → excluded from span
        assert_eq!(t.span_hit_rate(), Some((0.5, 2)));
    }

    #[test]
    fn parse_range_handles_valid_and_malformed() {
        assert_eq!(parse_line_range("40-80"), Some((40, 80)));
        assert_eq!(parse_line_range(" 5 - 5 "), Some((5, 5)));
        assert_eq!(parse_line_range("80-40"), None); // inverted
        assert_eq!(parse_line_range(""), None);
        assert_eq!(parse_line_range("abc"), None);
    }

    fn sr(file: &str, lines: [usize; 2]) -> SearchResult {
        SearchResult {
            chunk_id: format!("1-{file}"),
            file: std::path::PathBuf::from(file),
            lang: "rust".into(),
            symbol: None,
            kind: None,
            headings: Vec::new(),
            source: None,
            corpus_source: None,
            fetched: None,
            lines,
            score: 1.0,
            snippet: String::new(),
        }
    }

    #[test]
    fn ndcg_rewards_higher_ranked_relevant_and_credits_paths_once() {
        let expected = vec!["a.rs".to_string()];
        // Relevant at rank 0 → perfect nDCG.
        let top = vec![sr("a.rs", [1, 5]), sr("b.rs", [1, 5])];
        assert!((ndcg_at_10(&top, &expected) - 1.0).abs() < 1e-6);
        // Relevant at rank 1 → 1/log2(3) ≈ 0.6309.
        let mid = vec![sr("b.rs", [1, 5]), sr("a.rs", [1, 5])];
        assert!((ndcg_at_10(&mid, &expected) - 1.0 / 3f32.log2()).abs() < 1e-4);
        // No relevant result → 0.
        assert_eq!(ndcg_at_10(&[sr("z.rs", [1, 5])], &expected), 0.0);
    }

    #[test]
    fn span_hit_needs_path_and_overlap() {
        let expected = vec!["a.rs".to_string()];
        let results = vec![sr("a.rs", [40, 60])];
        assert!(span_hit(&results, &expected, (55, 90), 3)); // overlaps
        assert!(!span_hit(&results, &expected, (61, 90), 3)); // right file, disjoint lines
        let wrong = vec![sr("b.rs", [40, 60])];
        assert!(!span_hit(&wrong, &expected, (40, 60), 3)); // wrong file
    }
}
