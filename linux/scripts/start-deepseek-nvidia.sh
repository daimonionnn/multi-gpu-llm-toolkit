#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 entirely on the NVIDIA card - no AMD, no CPU experts.
#
# Only quants small enough to fit 96.6 GB usable VRAM alongside the MLA KV
# cache qualify (sizes from unsloth/DeepSeek-V4-Flash-0731-GGUF):
#
#   IQ2_M    85 GB    fits with 128k context  <- default (76 tg / 2118 pp measured)
#   IQ1_M    80.9 GB  fits with 200k context  <- --200k
#
# IQ quants are fine here: the HIP IQ-kernel fault is AMD-only and this profile
# never touches the AMD card. Quality warning: this is a QAT model (native
# ~4.25 bpw experts) and accuracy drops fast below ~3 bpw - these quants trade
# real quality for the single-card fit. For better quality use
# start-deepseek-nvidia-amd.sh or start-deepseek-mxfp4.sh.
#
# Extra variants with experts offloaded to system RAM (not pure-VRAM, but
# measured and verified - the fastest configs of the whole series):
#   --iq3       IQ3_XXS, 128k, 8 expert layers in RAM   (986 pp / 29 tg)
#   --iq3-256k  IQ3_XXS, 256k, 10 expert layers in RAM  (531 pp / 22 tg, verified)
#
# Usage:
#   ./start-deepseek-nvidia.sh                # IQ2_XXS, 128k, all in NVIDIA VRAM
#   ./start-deepseek-nvidia.sh --200k         # IQ1_M, 200k, all in NVIDIA VRAM
#   ./start-deepseek-nvidia.sh --iq3          # fastest (RAM-assisted)
#   ./start-deepseek-nvidia.sh --iq3-256k     # fastest at 256k (RAM-assisted)
#   ./start-deepseek-nvidia.sh -- --port 8090 # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

HF_DIR=/home/matt/.lmstudio/models/unsloth/DeepSeek-V4-Flash-0731-GGUF
MODEL_IQ2=$HF_DIR/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00002.gguf
MODEL_IQ1=$HF_DIR/DeepSeek-V4-Flash-0731-UD-IQ1_M-00001-of-00002.gguf
MODEL_IQ3=$HF_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf

require_model() {
    local first="$1" quant="$2"
    # Multi-part names vary by part count; accept any first part for the quant.
    local found
    found=$(ls "$HF_DIR"/*"$quant"*-00001-of-*.gguf 2>/dev/null | head -1)
    [[ -n "$found" ]] && { echo "$found"; return; }
    echo "Model not found: $quant in $HF_DIR" >&2
    echo "Download it first, e.g. in LM Studio search 'unsloth DeepSeek-V4-Flash-0731 $quant'" >&2
    exit 1
}

MODE_ARGS=(--mode cuda)
case "${1:-}" in
    --200k)
        shift
        MODEL=$(require_model "$MODEL_IQ1" UD-IQ1_M) || exit 1
        MODEL_ARGS=(-c 200000)
        ;;
    --iq3)
        shift
        MODEL=$(require_model "$MODEL_IQ3" UD-IQ3_XXS) || exit 1
        MODEL_ARGS=(-c 131072 --n-cpu-moe 8)
        ;;
    --iq3-256k)
        shift
        # Verified: two consecutive 261900-token prefills incl. a full-cache clear.
        MODEL=$(require_model "$MODEL_IQ3" UD-IQ3_XXS) || exit 1
        MODEL_ARGS=(-c 262144 --n-cpu-moe 10)
        ;;
    *)
        MODEL=$(require_model "$MODEL_IQ2" UD-IQ2_M) || exit 1
        MODEL_ARGS=(-c 131072)
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
