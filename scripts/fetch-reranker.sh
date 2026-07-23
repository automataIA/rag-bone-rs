#!/usr/bin/env bash
# Fetch a user-defined ONNX cross-encoder reranker into models/rerank/<name>/,
# where a CPU build resolves `reranker = "<name>"`. Picks the int8 variant matching
# the host CPU (avx512_vnni > avx512 > avx2 > arm64). Accelerated builds ignore
# this CPU prefetch unless `model-accelerated.onnx` is also provided. Model + 4 JSONs.
#
# Usage: scripts/fetch-reranker.sh [dest_root]   (default: ./models/rerank)
set -euo pipefail

NAME="ms-marco-MiniLM-L6-v2"
REPO="cross-encoder/ms-marco-MiniLM-L6-v2"
BASE="https://huggingface.co/${REPO}/resolve/main"
DEST_ROOT="${1:-models/rerank}"
DEST="${DEST_ROOT}/${NAME}"

# Architecture-specific int8 ONNX (all ~23 MB) from the repo's onnx/ dir.
flags="$(grep -m1 '^flags' /proc/cpuinfo || true)"
if   [[ "$flags" == *avx512_vnni* ]]; then VARIANT="model_qint8_avx512_vnni.onnx"
elif [[ "$flags" == *avx512f*     ]]; then VARIANT="model_qint8_avx512.onnx"
elif [[ "$flags" == *avx2*        ]]; then VARIANT="model_quint8_avx2.onnx"
elif [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then VARIANT="model_qint8_arm64.onnx"
else VARIANT="model.onnx"; fi  # fp32 fallback

echo "Fetching ${NAME} (${VARIANT}) → ${DEST}"
mkdir -p "$DEST"
curl -fsSL "${BASE}/onnx/${VARIANT}" -o "${DEST}/model.onnx"
for f in tokenizer.json config.json special_tokens_map.json tokenizer_config.json; do
  curl -fsSL "${BASE}/${f}" -o "${DEST}/${f}"
done

echo "Done. Set  reranker = \"${NAME}\"  in .rag-bone.toml (or --reranker ${NAME})."
ls -la "$DEST"
