#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 UD-IQ3_XXS (98 GB) entirely in VRAM across both cards.
#
# ############################################################################
# # THIS LAYOUT IS KNOWN TO CRASH. It is here because it is the fastest       #
# # generation measured on this rig for DeepSeek (57 t/s), not because it is  #
# # usable for anything that must finish.                                     #
# #                                                                           #
# # HSA_STATUS_ERROR_MEMORY_FAULT on the AMD card, somewhere between 43k and  #
# # 225k tokens of prefill work. The fault is probabilistic on the AMD expert #
# # path for every quant, but the HIP i-quant kernels hit it fastest, and     #
# # IQ3_XXS is an i-quant. Not fixed upstream as of 2026-08-11; the best      #
# # matching upstream bug (#26738) was investigated and ruled out. See        #
# # doc/benchmarks.md.                                                        #
# #                                                                           #
# # For unattended or fallback duty use start-deepseek-mxfp4-nvidia-cpu.sh,   #
# # which has never faulted. For IQ3_XXS specifically, the NVIDIA-only        #
# # variant is stable and *faster on prefill* despite spilling into RAM:      #
# # 936-986 pp against this layout's untestable number, because this one dies #
# # before a long prefill completes.                                          #
# #   ./start-deepseek-iq2xxs-nvidia.sh --iq3                                 #
# ############################################################################
#
# Budget: ~96.6 GB NVIDIA (all layers, KV, attention) + ~27 GB AMD (expert
# weights of 13 layers). Nothing in system RAM - hence the 57 t/s: the NVIDIA
# card runs attention while the AMD card serves expert matmuls in parallel,
# instead of the two taking turns.
#
# Usage:
#   ./start-deepseek-iq3xxs-nvidia-amd.sh                # 128k (default)
#   ./start-deepseek-iq3xxs-nvidia-amd.sh --200k         # the full window
#   ./start-deepseek-iq3xxs-nvidia-amd.sh -- --port 8090 # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

HF_DIR=/home/matt/.lmstudio/models/unsloth/DeepSeek-V4-Flash-0731-GGUF

# Multi-part names vary by part count; accept any first part for the quant.
require_model() {
    local quant="$1" found
    found=$(ls "$HF_DIR"/*"$quant"*-00001-of-*.gguf 2>/dev/null | head -1)
    [[ -n "$found" ]] && { echo "$found"; return; }
    echo "Model not found: $quant in $HF_DIR" >&2
    echo "Download it first, e.g. in LM Studio search 'unsloth DeepSeek-V4-Flash-0731 $quant'" >&2
    exit 1
}

# Experts of layers 30-42 (13 layers, ~26-27 GB) on the AMD card.
OT_PATTERN='blk\.(3[0-9]|4[0-2])\.ffn_.*_exps.*=ROCm0'

CTX=131072
if [[ "${1:-}" == "--200k" ]]; then
    shift
    CTX=200000
fi
[[ "${1:-}" == "--" ]] && shift

MODEL=$(require_model UD-IQ3_XXS) || exit 1

echo "WARNING: IQ3_XXS experts on the AMD card fault intermittently (HIP i-quant kernels)." >&2
echo "         Expect HSA_STATUS_ERROR_MEMORY_FAULT within ~43k-225k tokens of prefill." >&2
echo "         Stable alternatives: start-deepseek-mxfp4-nvidia-cpu.sh (fallback duty)," >&2
echo "         or ./start-deepseek-iq2xxs-nvidia.sh --iq3 (same quant, NVIDIA + RAM)." >&2

exec "$SCRIPT_DIR/start-llama-server.sh" --mode rocm-cuda \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    -c "$CTX" \
    -ts 0,1 -ot "$OT_PATTERN" \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
