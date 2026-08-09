#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 UD-IQ3_XXS (98 GB, MoE 256 experts, MLA).
#
# Default is CUDA-only with 8 layers of experts in system RAM - measured
# stable through repeated 130k-token prefills and full-cache clears.
#
# The dual rocm-cuda mode generates ~1.4x faster but the DSV4 KV-cache code
# faults intermittently on the ROCm backend (HSA_STATUS_ERROR_MEMORY_FAULT
# during long prefills or after clearing a large cache; reproduced at 43k,
# ~130k and 256k scales). DSV4 support in llama.cpp is days old - retest dual
# after upstream updates. Details: doc/benchmarks.md, DeepSeek section.
#
# Usage:
#   ./start-deepseek-v4-flash.sh                 # stable: CUDA-only, 128k ctx
#   ./start-deepseek-v4-flash.sh --256k          # stable: CUDA-only, full 256k ctx
#   ./start-deepseek-v4-flash.sh --dual          # faster tg, UNSTABLE today
#   ./start-deepseek-v4-flash.sh -- --port 8090  # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

MODEL=/home/matt/.lmstudio/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf

MODE_ARGS=(--mode cuda)
MODEL_ARGS=(-c 131072 --n-cpu-moe 8)
if [[ "${1:-}" == "--256k" ]]; then
    shift
    # Verified: two consecutive 261900-token prefills incl. a full-cache clear.
    MODEL_ARGS=(-c 262144 --n-cpu-moe 10)
elif [[ "${1:-}" == "--dual" ]]; then
    shift
    echo "WARNING: dual rocm-cuda is unstable for this model (intermittent ROCm faults)." >&2
    MODE_ARGS=(--mode rocm-cuda)
    MODEL_ARGS=(-c 262144)
fi
[[ "${1:-}" == "--" ]] && shift

exec "$SCRIPT_DIR/start-llama-server.sh" "${MODE_ARGS[@]}" \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    "${MODEL_ARGS[@]}" \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
