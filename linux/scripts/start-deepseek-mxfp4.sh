#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 MXFP4 (146 GB) - the lossless reference quant.
#
# The model is QAT with native MXFP4 experts, so this quant IS the original
# weights; every other quant is equal at best (Q8) or lossy (Q4 and below).
# Too large for VRAM alone: expert-offload dual plus experts of the first
# 10-12 layers in system RAM. Gauntlet-verified stable on the AMD card
# (MXFP4 is not an IQ quant, so the HIP IQ-kernel fault does not apply).
#
# Sibling profiles: start-deepseek-nvidia.sh (single-card fits),
# start-deepseek-nvidia-amd.sh (all-VRAM dual). Numbers: doc/benchmarks.md.
#
# Usage:
#   ./start-deepseek-mxfp4.sh                 # full 256k context (default)
#   ./start-deepseek-mxfp4.sh --128k          # 128k, two fewer expert layers in RAM
#   ./start-deepseek-mxfp4.sh --cuda-only     # no AMD card: slower (pp ~305, tg ~16.4)
#                                             # but immune to the ROCm faults - use
#                                             # for unattended/fallback service duty
#   ./start-deepseek-mxfp4.sh -- --port 8090  # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

MODEL=/home/matt/.lmstudio/models/lmstudio-community/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf

# Default: full 256k. KV needs ~13 GB on CUDA0, so experts of the first 12
# layers go to system RAM; the AMD card serves expert matmuls for 8 layers.
# 256k-verified: 386 pp / 18.4 tg on a full 261900-token prompt, incl. the
# clear-and-refill cycle.
MODE_ARGS=(--mode rocm-cuda)
OT_ARGS=(-ts 0,1 -ot 'blk\.(3[5-9]|4[0-2])\.ffn_.*_exps.*=ROCm0')
MODEL_ARGS=(-c 262144 --n-cpu-moe 12)
if [[ "${1:-}" == "--128k" ]]; then
    shift
    # Gauntlet-verified: 480-592 pp, 21-25 tg.
    MODEL_ARGS=(-c 131072 --n-cpu-moe 10)
elif [[ "${1:-}" == "--cuda-only" ]]; then
    shift
    # The AMD expert path faults intermittently on ALL quants (reproduced on
    # MXFP4 in production at ~45k prefill tokens, 2026-08-09, despite passing
    # the gauntlets - low-rate, probabilistic). CUDA-only never faulted once.
    MODE_ARGS=(--mode cuda)
    OT_ARGS=()
    MODEL_ARGS=(-c 131072 --n-cpu-moe 18)
fi
[[ "${1:-}" == "--" ]] && shift

exec "$SCRIPT_DIR/start-llama-server.sh" "${MODE_ARGS[@]}" \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    "${MODEL_ARGS[@]}" \
    ${OT_ARGS[@]+"${OT_ARGS[@]}"} \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
