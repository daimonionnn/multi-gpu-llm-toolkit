#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 UD-IQ3_XXS (98 GB, MoE 256 experts, MLA) - dual GPU.
#
# MLA keeps the KV cache tiny (~50 KB/token across all 43 layers), so 256k
# context fits: ~98 GB weights + ~13 GB KV across 96 + 32 GB of VRAM.
# Requires the CUDA 12.8 runtime (build-cuda12-container.sh) - see
# doc/cuda-fa-blackwell.md for why the 13.3 build must not be used.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

MODEL=/home/matt/.lmstudio/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf

MODE="rocm-cuda"
if [[ "${1:-}" == "--mode" ]]; then
    MODE="${2:?}"; shift 2
fi

exec "$SCRIPT_DIR/start-llama-server.sh" --mode "$MODE" \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    -c 262144 \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
