#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 entirely in VRAM across both cards - no CPU experts.
#
# Budget: ~96.6 GB NVIDIA (all layers, KV, attention) + ~27 GB AMD (expert
# weights of 13 layers) = weights up to ~110 GB with 200k context.
#
#   Q2_K_XL  90.2 GB  <- default: the only quant that fits AND is stable on
#                        the AMD card (k-quant; the HIP fault hits IQ quants)
#   IQ3_XXS  97.1 GB  <- --iq3: better quality, fits, measured 57 t/s - but
#                        UNSTABLE: HIP IQ-kernels fault intermittently
#
# Details and the stability matrix: doc/benchmarks.md, DeepSeek section.
#
# Usage:
#   ./start-deepseek-nvidia-amd.sh                # Q2_K_XL, 200k, all-VRAM dual
#   ./start-deepseek-nvidia-amd.sh --iq3          # IQ3_XXS, 200k - UNSTABLE
#   ./start-deepseek-nvidia-amd.sh -- --port 8090 # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

HF_DIR=/home/matt/.lmstudio/models/unsloth/DeepSeek-V4-Flash-0731-GGUF

require_model() {
    local quant="$1" found
    found=$(ls "$HF_DIR"/*"$quant"*-00001-of-*.gguf 2>/dev/null | head -1)
    [[ -n "$found" ]] && { echo "$found"; return; }
    echo "Model not found: $quant in $HF_DIR" >&2
    echo "Download it first, e.g. in LM Studio search 'unsloth DeepSeek-V4-Flash-0731 $quant'" >&2
    exit 1
}

# All layers, KV and attention on CUDA0; the AMD card serves expert matmuls
# for 13 layers (~26-27 GB) from VRAM. Nothing in system RAM.
OT_PATTERN='blk\.(3[0-9]|4[0-2])\.ffn_.*_exps.*=ROCm0'

if [[ "${1:-}" == "--iq3" ]]; then
    shift
    echo "WARNING: IQ3_XXS experts on the AMD card fault intermittently (HIP IQ kernels)." >&2
    echo "         Expect a crash somewhere within ~50k-250k tokens of prefill work." >&2
    MODEL=$(require_model UD-IQ3_XXS) || exit 1
else
    MODEL=$(require_model UD-Q2_K_XL) || exit 1
fi
[[ "${1:-}" == "--" ]] && shift

exec "$SCRIPT_DIR/start-llama-server.sh" --mode rocm-cuda \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    -c 200000 \
    -ts 0,1 -ot "$OT_PATTERN" \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
