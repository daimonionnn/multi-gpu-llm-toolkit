#!/usr/bin/env bash
# gpt-oss-120b MXFP4 (60 GB, MoE 128/4, native MXFP4) - CUDA-only, 128k ctx.
#
# The fastest model measured on this rig: 258 tg / 9950 pp at short context,
# 148 tg on a 126k-token prompt. Fits the NVIDIA card whole; the dual layout
# only slows it 3x, so there is none here. Numbers: doc/benchmarks.md.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

# GGUF root; override with LLM_MODELS_DIR. Default is LM Studio's download dir.
MODELS_DIR="${LLM_MODELS_DIR:-$HOME/.lmstudio/models}"

MODEL="$MODELS_DIR/lmstudio-community/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4-00001-of-00002.gguf"
[[ -f "$MODEL" ]] || {
    echo "Model not found: $MODEL" >&2
    echo "Download it first, e.g. in LM Studio search 'lmstudio-community gpt-oss-120b MXFP4'" >&2
    echo 'Or set LLM_MODELS_DIR if your GGUFs live outside $HOME/.lmstudio/models.' >&2
    exit 1
}

[[ "${1:-}" == "--" ]] && shift
exec "$SCRIPT_DIR/start-llama-server.sh" --mode cuda \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    -c 131072 \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
