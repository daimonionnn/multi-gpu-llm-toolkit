#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 entirely in VRAM across both cards - no CPU experts.
#
# Budget: ~96.6 GB NVIDIA (all layers, KV, attention) + ~27 GB AMD (expert
# weights of 13 layers) = weights up to ~110 GB with 200k context.
#
#   Q2_K_XL  90.2 GB  <- default: the only quant that fits AND has the lower
#                        fault rate on the AMD card (the HIP fault is
#                        probabilistic on all quants - IQ quants fail fast,
#                        non-IQ rarely but not never; see benchmarks.md)
#   IQ3_XXS  97.1 GB  <- start-deepseek-iq3xxs-nvidia-amd.sh: better quality
#                        and 57 t/s, but UNSTABLE (HIP i-quant kernels).
#                        `--iq3` here still works, it just redirects there.
#
# Details and the stability matrix: doc/benchmarks.md, DeepSeek section.
#
# Usage:
#   ./start-deepseek-q2kxl-nvidia-amd.sh                # Q2_K_XL, 128k, all-VRAM dual
#   ./start-deepseek-q2kxl-nvidia-amd.sh --200k         # Q2_K_XL at the full 200k
#   ./start-deepseek-q2kxl-nvidia-amd.sh --iq3          # redirects to the IQ3_XXS profile
#   ./start-deepseek-q2kxl-nvidia-amd.sh -- --port 8090 # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

# GGUF root; override with LLM_MODELS_DIR. Default is LM Studio's download dir.
MODELS_DIR="${LLM_MODELS_DIR:-$HOME/.lmstudio/models}"

HF_DIR="$MODELS_DIR/unsloth/DeepSeek-V4-Flash-0731-GGUF"

require_model() {
    local quant="$1" found
    found=$(ls "$HF_DIR"/*"$quant"*-00001-of-*.gguf 2>/dev/null | head -1)
    [[ -n "$found" ]] && { echo "$found"; return; }
    echo "Model not found: $quant in $HF_DIR" >&2
    echo "Download it first, e.g. in LM Studio search 'unsloth DeepSeek-V4-Flash-0731 $quant'" >&2
    echo 'Or set LLM_MODELS_DIR if your GGUFs live outside $HOME/.lmstudio/models.' >&2
    exit 1
}

# All layers, KV and attention on CUDA0; the AMD card serves expert matmuls
# for 13 layers (~26-27 GB) from VRAM. Nothing in system RAM.
OT_PATTERN='blk\.(3[0-9]|4[0-2])\.ffn_.*_exps.*=ROCm0'

# 128k default, matching the other profiles. 200k still fits - it was the
# original default - so the smaller window is a consistency choice, not a
# capacity one. Everything is in VRAM here, so the freed KV space is simply
# headroom; it was NOT used to move experts back off the AMD card, because that
# retune has not been measured.
CTX=131072
if [[ "${1:-}" == "--200k" ]]; then
    shift
    CTX=200000
fi

# --iq3 used to be handled here, but a script named for one quant serving a
# different one contradicts the start-<model>-<quant>-<placement> naming. It now
# lives in its own profile; the flag stays as a redirect so it does not break.
if [[ "${1:-}" == "--iq3" ]]; then
    shift
    exec "$SCRIPT_DIR/start-deepseek-iq3xxs-nvidia-amd.sh" "$@"
fi
MODEL=$(require_model UD-Q2_K_XL) || exit 1
[[ "${1:-}" == "--" ]] && shift

exec "$SCRIPT_DIR/start-llama-server.sh" --mode rocm-cuda \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    -c "$CTX" \
    -ts 0,1 -ot "$OT_PATTERN" \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
