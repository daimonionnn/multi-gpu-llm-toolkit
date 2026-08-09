#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 (MoE 256 experts, MLA) - profile with two quants:
#
#   IQ3_XXS (98 GB)  - fastest; fits the NVIDIA card, CUDA-only. Lowest quality.
#   MXFP4  (146 GB)  - lossless reference (the model is QAT with native MXFP4
#                      experts); expert-offload dual, stable through gauntlets.
#
# The classic layer-split dual is unstable for the IQ3 quant only: the HIP
# IQ-series MoE kernels fault intermittently on gfx1201. MXFP4/k-quant expert
# paths on the AMD card are fine (~0.5M prefill tokens without a fault).
# Full numbers and the stability matrix: doc/benchmarks.md, DeepSeek section.
#
# Usage:
#   ./start-deepseek-v4-flash.sh                 # MXFP4, expert-offload dual, 128k - best quality (default)
#   ./start-deepseek-v4-flash.sh --iq3           # IQ3_XXS, CUDA-only, 128k - fastest
#   ./start-deepseek-v4-flash.sh --256k          # IQ3_XXS, CUDA-only, full 256k
#   ./start-deepseek-v4-flash.sh --dual          # IQ3_XXS classic dual - UNSTABLE (IQ kernels)
#   ./start-deepseek-v4-flash.sh -- --port 8090  # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

MODEL_IQ3=/home/matt/.lmstudio/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
MODEL_MXFP4=/home/matt/.lmstudio/models/lmstudio-community/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf

# Default: the lossless quant on the expert-offload dual. Every layer, the KV
# cache and attention stay on CUDA0 (-ts 0,1); the AMD card serves expert
# matmuls for 8 layers from VRAM; experts of the first 10 layers go to system
# RAM. Gauntlet-verified stable; 480-592 pp, 21-25 tg.
MODEL="$MODEL_MXFP4"
MODE_ARGS=(--mode rocm-cuda)
MODEL_ARGS=(-c 131072 --n-cpu-moe 10
            -ts 0,1 -ot 'blk\.(3[5-9]|4[0-2])\.ffn_.*_exps.*=ROCm0')
case "${1:-}" in
    --iq3)
        shift
        MODEL="$MODEL_IQ3"
        MODE_ARGS=(--mode cuda)
        MODEL_ARGS=(-c 131072 --n-cpu-moe 8)
        ;;
    --256k)
        shift
        # IQ3 CUDA-only is the verified 256k path: two consecutive
        # 261900-token prefills incl. a full-cache clear.
        MODEL="$MODEL_IQ3"
        MODE_ARGS=(--mode cuda)
        MODEL_ARGS=(-c 262144 --n-cpu-moe 10)
        ;;
    --dual)
        shift
        echo "WARNING: classic dual is unstable with the IQ3 quant (HIP IQ-kernel faults)." >&2
        echo "         The default (no flag) is the stable MXFP4 expert-offload dual." >&2
        MODE_ARGS=(--mode rocm-cuda)
        MODEL_ARGS=(-c 262144)
        ;;
esac
[[ "${1:-}" == "--" ]] && shift

exec "$SCRIPT_DIR/start-llama-server.sh" "${MODE_ARGS[@]}" \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    "${MODEL_ARGS[@]}" \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
