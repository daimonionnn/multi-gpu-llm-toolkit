#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 entirely on the NVIDIA card - no AMD, no CPU experts.
#
# Default: antirez IQ2XXS+Q8-attn imatrix (80.8 GB). Asymmetric recipe -
# routed experts at IQ2_XXS/Q2_K but attention projections, shared experts,
# router, output head and embeddings kept at Q8/F16, with an imatrix
# calibrated on chat-v2 traffic. Measured better than unsloth's evenly-spread
# UD-IQ2_M despite being smaller (PPL 6.08 vs 7.09, KLD 0.408 vs 0.484, top
# token 78.2% vs 76.6%), and the bits sit exactly where agent work needs them:
# routing and tool-call decisions. Also leaves room for ~200k context.
#
#   antirez IQ2XXS+Q8  80.8 GB  128k ctx  <- default (2007 pp / 74.5 tg)
#   unsloth UD-IQ2_M   85 GB    128k ctx  <- --iq2m (2118 pp / 76.1 tg)
#   unsloth UD-IQ1_M   80.9 GB  200k ctx  <- --200k (untested, lowest quality)
#
# IQ quants are safe here: the HIP IQ-kernel fault is AMD-only and this
# profile never touches the AMD card. Quality caveat: all of these are ~2-bit
# re-encodes of a QAT model whose experts are natively ~4.25 bpw, so ~1 token
# in 5 differs from the Q8 reference. For quality use start-deepseek-mxfp4-nvidia-amd-cpu.sh;
# for raw speed with full model fidelity consider start-gptoss-mxfp4-nvidia.sh.
#
# IQ3_XXS with experts in system RAM now lives in its own profile,
# start-deepseek-iq3xxs-nvidia-cpu.sh (986 pp / 29 tg at 128k). The --iq3 and
# --iq3-256k flags here still work; they redirect there.
#
# Usage:
#   ./start-deepseek-iq2xxs-nvidia.sh                # antirez IQ2XXS+Q8, 128k (default)
#   ./start-deepseek-iq2xxs-nvidia.sh --200k         # IQ1_M, 200k
#   ./start-deepseek-iq2xxs-nvidia.sh --iq2m         # unsloth UD-IQ2_M, 128k
#   ./start-deepseek-iq2xxs-nvidia.sh --iq3          # redirects to the IQ3_XXS profile
#   ./start-deepseek-iq2xxs-nvidia.sh --iq3-256k     # the same, at 256k
#   ./start-deepseek-iq2xxs-nvidia.sh -- --port 8090 # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

HF_DIR=/home/matt/.lmstudio/models/unsloth/DeepSeek-V4-Flash-0731-GGUF
ANTIREZ_DIR=/home/matt/.lmstudio/models/antirez/deepseek-v4-gguf
MODEL_ANTIREZ=$ANTIREZ_DIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

# Multi-part names vary by part count; accept any first part for the quant.
require_unsloth() {
    local quant="$1" found
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
        MODEL=$(require_unsloth UD-IQ1_M) || exit 1
        MODEL_ARGS=(-c 200000)
        ;;
    --iq2m)
        shift
        MODEL=$(require_unsloth UD-IQ2_M) || exit 1
        MODEL_ARGS=(-c 131072)
        ;;
    # IQ3_XXS moved to its own profile - a script named for one quant should
    # not serve another. The flags stay as redirects so nothing breaks.
    --iq3)
        shift
        exec "$SCRIPT_DIR/start-deepseek-iq3xxs-nvidia-cpu.sh" "$@"
        ;;
    --iq3-256k)
        shift
        exec "$SCRIPT_DIR/start-deepseek-iq3xxs-nvidia-cpu.sh" --256k "$@"
        ;;
    *)
        MODEL="$MODEL_ANTIREZ"
        [[ -f "$MODEL" ]] || { echo "Model not found: $MODEL" >&2; exit 1; }
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
