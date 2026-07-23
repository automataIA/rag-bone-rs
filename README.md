<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-rag-bone-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-rag-bone-light.svg">
    <img alt="rag-bone-rs logo" src="assets/logo-rag-bone-light.svg" width="200">
  </picture>
</p>

# rag-bone-rs

High-performance Rust CLI for **semantic search** over documentation (Markdown) and
multi-language codebases. It indexes via **embeddings + chunking** and returns few,
**highly relevant** results through **hybrid dense + BM25** retrieval fused with RRF
(optional cross-encoder, off by default — see [Performance](#performance--recommended-usage)).
The CPU build is a single binary; accelerated builds also ship the provider's ONNX Runtime
dynamic libraries. No daemons: it starts, searches, exits. JSON output for agents, text for
humans. For query-heavy workloads there is also a streaming mode that keeps the models
loaded in the same process.

## Commands

```bash
rag-bone config --init                       # create .rag-bone.toml
rag-bone index  --path ./docs --path ./src   # index (incremental)
rag-bone search "how does X work" --json --limit 3
rag-bone search "error E0425" --retrieval bm25 --json --limit 3 # BM25: zero models
rag-bone search "how does X work" --compact  # chunk_id + one-line preview (for agents)
rag-bone get   1-<hash> --context-lines 5    # open a chunk by id, no models loaded
rag-bone outline src/store.rs                # structural map of a file (no models)
rag-bone find  save                          # where a symbol is defined (no models)
rag-bone catalog                             # indexed web sources: URL, fetch date, chunk count (no models)
rag-bone search "install" --source serde.rs  # search scoped to one web source from the catalog
printf '%s\n' "query one" "query two" | rag-bone search-batch  # JSONL, long-lived models
rag-bone status                              # model, dim, file/chunk counts, provider
rag-bone watch  --path ./src                 # incremental reindex on changes
```

Metadata-first agent contract: `search --compact` returns `chunk_id`, path, lang, symbol,
Markdown breadcrumb, web provenance, line range, score and a **one-line preview** instead of
the full snippet; `get <chunk_id>` then fetches the full text (`--context-lines N` widens it
by reading the file) without reloading models. `outline <file>` lists the chunks of a file
(symbol/kind/span/signature) and `find <symbol>` locates where a symbol is defined (metadata,
with a BM25 lexical fallback) — both without loading models.

During `index` and `eval`, phases and progress every ~5% go to **stderr** (`percent`,
completed units and total); the initial model download shows its own progress bar. stdout
stays clean for text, JSON and JSONL results. The percentage counts completed files/queries
and is not an estimate of remaining time.

### Search — flags

| Flag | Effect |
|---|---|
| `--limit N` / `-k N` | maximum number of results (default 3) |
| `--json` | JSON array on stdout (logs on stderr) |
| `--compact` | agent form: `chunk_id` + one-line preview instead of the full snippet |
| `--retrieval dense\|bm25\|hybrid` | retrieval channel (default `hybrid`) |
| `--lang rust,md` | restrict to languages |
| `--path-prefix src/` | only paths containing the substring |
| `--no-rerank` | disable the cross-encoder (already off by default: hybrid without reranker) |
| `--reranker <id>` | reranker override (built-in id, user-defined name, or path to a `.onnx`) |

`bm25` is genuinely models-free: it neither initializes nor downloads the embedder.
`search-batch` accepts the same filter/rerank flags and reads one non-empty query per line
from stdin. It emits one `{"query": ..., "results": [...]}` object per line and loads the
embedding model and reranker once: it is the recommended path for CPU throughput.

Shape of a JSON result (`symbol`/`kind` fields are omitted for prose):

```json
{ "chunk_id": "1-<hash>", "file": "src/auth.rs", "lang": "rust", "symbol": "verify_token",
  "kind": "function", "headings": [], "source": "https://requested.example/docs",
  "corpus_source": "https://requested.example/llms-full.txt", "fetched": "2026-07-22T08:00:00Z",
  "lines": [42, 88], "score": 0.83, "snippet": "fn verify_token(...) { ... }" }
```

With `--compact` the `snippet` is replaced by a one-line `preview`; use the `chunk_id` with
`rag-bone get` afterwards.

## Performance / recommended usage

The **defaults already are the maximum-performance configuration** (quality *and* speed):
`hybrid` retrieval with reranker **off** wins on every metric of the measured golden set
**and** is ~50× faster than the cross-encoder path. Numbers and methodology in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

**Recipe:**

- **Build**
  - CPU (default, portable single binary): `cargo build --release` — ~18 ms per query with models loaded.
  - GPU (CUDA): `cargo build --release --features gpu` — mostly useful for **indexing** (reindex
    17–21× faster: ~11 s vs 234 s on 1239 chunks). With the reranker off, the GPU gain on *queries* is
    marginal (~1.2×), so the GPU pays off for large repos / frequent reindexing, not for latency.
- **Config**: keep the defaults (`retrieval = "hybrid"`, `rerank = false`, `retrieve_n = 30`,
  `model = "embeddinggemma-300m-q4"`). `retrieve_n` at 50/100 adds no quality, only latency.
- **Many queries / agent session**: use `rag-bone search-batch` — it loads the models **once**
  and avoids the cold start (~1.4–2.3 s per one-shot invocation, dominated by model load).
- **Token-cheap discovery**: `search --compact` (chunk_id + preview) → then `get <chunk_id>`.
- **Instant navigation, ZERO models loaded**: BM25, `get`, `find <symbol>`, `outline <file>`.
- **Agent protocol**: `search --compact` → `find`/`outline` on the symbols that surfaced →
  targeted `get`; 2–3 short iterations instead of injecting many similar chunks.
- **Reranker**: re-enable it only if `eval` on your corpus justifies it (see above).
- **Do not raise** `MAX_TOKENS` (512) or the batch size (16): they are load-bearing memory caps.

## Configuration (`.rag-bone.toml`)

```toml
sources      = ["./docs", "./src"]
model        = "embeddinggemma-300m-q4"
retrieval    = "hybrid"                  # dense | bm25 | hybrid (dense + BM25 via RRF)
rerank       = false                     # cross-encoder off: on the measured corpus it hurts and costs ~50×
reranker     = "ms-marco-MiniLM-L6-v2"   # used only if rerank = true
rrf_k        = 60                         # Reciprocal Rank Fusion constant (hybrid only)
top_k        = 3
retrieve_n   = 30
min_score    = 0.0
max_per_file = 0                          # cap chunks per file after ranking (0 = unlimited)

[chunk]
max_chars = 1500
overlap   = 0
```

### Reranker

The cross-encoder is **off by default** (`rerank = false`): on the measured corpus it demotes
the right chunk out of the hybrid candidate set and is ~50× slower. Re-enable it
(`rerank = true`, or by dropping `--no-rerank`) only if `rag-bone eval` on *your* corpus shows
it helps. When active:

`reranker` accepts: a **built-in id** (`bge-reranker-base`, `jina-reranker-v1-turbo-en`,
`bge-reranker-v2-m3`, `jina-reranker-v2-base-multilingual`), a **user-defined name** resolved
to `models/rerank/<name>/model.onnx`, or a **path** to an arbitrary `.onnx`. Relative paths
are resolved against the config file's directory.

The default `ms-marco-MiniLM-L6-v2` **auto-downloads on first use** via `hf-hub` into the
shared cache (`~/.cache/huggingface`). The CPU build picks the int8 for the architecture
(avx512_vnni / avx512 / avx2 / arm64); accelerated builds pick FP32, a much better fit for
the GPU.

Resolution order: `.onnx` path → built-in id → HF/provider-specific recipe → other local
prefetch. For known recipes, CPU uses `models/rerank/<name>/model.onnx`; an accelerator uses
`model-accelerated.onnx` if present, otherwise it downloads FP32. For offline CPU prefetch:

```bash
scripts/fetch-reranker.sh          # → models/rerank/ms-marco-MiniLM-L6-v2/ (optional, takes priority)
```

For maximum quality use `bge-reranker-base` (built-in, slower). See `docs/RESEARCH.md`.

## Architecture

- **Embeddings**: `fastembed` (ONNX, download-once, offline). Configurable model.
- **Chunking**: `text-splitter` — Markdown-aware for docs, tree-sitter (function/class
  boundaries) for code.
- **Storage**: versioned flat file `.rag-bone/index.bin` (postcard/Serde streaming), validated
  on read and before saving. Written via temporary file with atomic replace on Unix; cosine
  search in RAM parallelized with `rayon` — exact, not approximate.
- **Incrementality**: per-file content hash plus a pipeline fingerprint (schema, model,
  prefixes and chunking); changes that invalidate the vectors automatically force a rebuild.
- **Retrieval**: hybrid dense (exact cosine) + in-memory lexical BM25, fused with Reciprocal
  Rank Fusion; selectable channels (`retrieval = dense|bm25|hybrid`). BM25 is derived from the
  records at load time, so it is not a persisted artifact and always stays in sync with the
  vectors.
- **Rerank**: optional cross-encoder `ms-marco-MiniLM-L6-v2` (int8 on CPU, FP32 on
  accelerator), **off by default** — on the measured corpus it degrades hybrid retrieval;
  re-enable with `rerank = true`.

### Acceleration

- **Fail-fast acceleration**: build with exactly one of `gpu` (CUDA on Linux/WSL2), `directml`
  (Windows) and `coreml` (macOS). If the provider fails to register, model loading fails
  instead of pretending a GPU build on CPU; individual unsupported operators may still fall
  back to CPU.
- **Parallelism**: `rayon` for vector scoring; embedding and rerank use ONNX batches.
- **Optimized, portable binary**: `cargo build --release` uses fat LTO without pinning
  `target-cpu=native`. Host-specific CPU optimizations should be enabled for local builds only.

## Build

Requires Rust **1.88 or later**.

```bash
cargo build --release                 # CPU
cargo build --release --features gpu  # CUDA strict + ORT provider dylibs
cargo build --release --features directml  # Windows / DirectML
cargo build --release --features coreml    # macOS / CoreML
```

The acceleration features are mutually exclusive. The resulting binary is
`target/release/rag-bone`; `ort/copy-dylibs` places the provider libraries next to it. When
distributing the GPU build, also copy (dereferencing links) at least
`libonnxruntime_providers_shared.so` and `libonnxruntime_providers_cuda.so`. First use
downloads the models into the fastembed/Hugging Face caches.

Release benchmark on Ryzen 5 7600 + RTX 4070, 1239 chunks / 45 queries: CPU→CUDA goes from
778→69 ms with MiniLM, 1247→75 ms with Jina and 6136→202 ms with BGE. The default reindex
goes from 234.29 s to 10.89–13.74 s. Details and methodology in `docs/RESEARCH.md`.

The research roadmap for Sirbone favors structural headers in chunks, hybrid dense+BM25/RRF
retrieval, per-chunk embedding cache and a metadata-first agent output with targeted reads.
The flat index intentionally stays unchanged until larger-scale benchmarks show that scanning
or saving is the bottleneck.

> **Index format migration (schema 4):** older indexes do not include `corpus_source`.
> Recreate them once with `rag-bone index --reindex`; subsequent updates are incremental
> again.

## As a skill (Claude Code / sir-bone / any Agent Skills client)

The skill ships in two variants with an identical body:

- `skills/rag-bone/SKILL.md` — [Agent Skills](https://agentskills.io) open standard
  (sir-bone, OpenCode, pi, …);
- `skills/claude-code/rag-bone/SKILL.md` — Claude Code extensions (`when_to_use`,
  `allowed-tools`).

Install straight from the repository:

```bash
# Claude Code (plugin marketplace — installs the claude-code variant)
/plugin marketplace add automataIA/rag-bone-rs
/plugin install rag-bone@rag-bone

# any Agent Skills client (open-standard variant, 70+ agents)
npx skills add automataIA/rag-bone-rs

# GitHub Copilot CLI
gh skill install automataIA/rag-bone-rs
```

Or from a local clone:

- Claude Code: `scripts/install-skill.sh --claude`
- sir-bone: `scripts/install-skill.sh --sirbone` (default)

Use `--check` with the target to detect out-of-sync copies without modifying them, e.g.
`scripts/install-skill.sh --check --claude`.

It instructs the agent to use grep/find for exact lookups and BM25/hybrid for lexical or
conceptual retrieval, following the token-frugal `compact → get` protocol.

## Parameter tuning (autoresearch)

`eval/` contains the scaffold for autonomous, autoresearch-style tuning: sweeps over model,
chunking rules and retrieval parameters against a golden set, with Recall@k / MRR metrics.
See `eval/program.md`.

## License

Dual-licensed under either of:

- [Apache License 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Any contribution intentionally submitted for inclusion in this project shall be dual-licensed
as above, without any additional terms or conditions.
