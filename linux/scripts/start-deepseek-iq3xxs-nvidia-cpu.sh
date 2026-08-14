#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 UD-IQ3_XXS (98 GB) on the NVIDIA card, experts of the
# first few layers in system RAM. No AMD card, so none of the HIP faults apply.
#
# This is the stable way to run IQ3_XXS, and the counter-intuitive one: it is
# *faster on prefill* than putting the same quant entirely in VRAM across both
# cards. 936-986 pp here against a dual layout that generates at 57 t/s but
# dies of HSA_STATUS_ERROR_MEMORY_FAULT within 43k-225k prefill tokens
# (start-deepseek-iq3xxs-nvidia-amd.sh) - a number you cannot collect if the
# run does not finish.
#
#   128k, 8 expert layers in RAM   986 pp / 29.3 tg   <- default
#   256k, 10 expert layers in RAM  531 pp / 22.0 tg   <- --256k, verified with
#                                  two consecutive 261900-token prefills
#                                  including a full-cache clear
#
# Quality note: at ~3 bpw this sits below MXFP4, which is the QAT model's
# native expert format and therefore lossless. QAT models lose accuracy faster
# than BF16-trained ones once experts drop under ~3 bpw, so IQ3_XXS is the
# floor rather than a middle option. For quality use
# start-deepseek-mxfp4-nvidia-amd-cpu.sh; for the all-VRAM 2-bit speed point
# use start-deepseek-iq2xxs-nvidia.sh.
#
# Usage:
#   ./start-deepseek-iq3xxs-nvidia-cpu.sh                # 128k (default)
#   ./start-deepseek-iq3xxs-nvidia-cpu.sh --256k         # 256k, 10 layers in RAM
#   ./start-deepseek-iq3xxs-nvidia-cpu.sh -- --port 8090 # extra llama-server args

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

# -b 4096 -ub 2048 for the same reason as the MXFP4 profile: with experts in
# host memory, the micro-batch size decides how well each weight transfer to
# the GPU is amortised. Measured there at +55-60% prefill; carried over here by
# the same mechanism rather than by its own measurement.
#
# --no-mmap because experts are overridden to the CPU, which is exactly the
# case llama.cpp warns about at startup ("tensor overrides to CPU are used with
# mmap enabled - consider using --no-mmap for better performance").
MODEL_ARGS=(-c 131072 --n-cpu-moe 8 --no-mmap)
if [[ "${1:-}" == "--256k" ]]; then
    shift
    # The bigger KV cache costs two more expert layers to system RAM, and the
    # prefill rate nearly halves with them - the RAM path is the bottleneck.
    MODEL_ARGS=(-c 262144 --n-cpu-moe 10 --no-mmap)
fi
[[ "${1:-}" == "--" ]] && shift

MODEL=$(require_model UD-IQ3_XXS) || exit 1

exec "$SCRIPT_DIR/start-llama-server.sh" --mode cuda \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    "${MODEL_ARGS[@]}" \
    -ngl 99 \
    -fa on \
    -b 4096 -ub 2048 \
    --jinja \
    "$@"
