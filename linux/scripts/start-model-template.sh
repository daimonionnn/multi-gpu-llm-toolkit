#!/usr/bin/env bash
# Model profile template — copy this per model, as the Windows side does with
# start-qwen122b-q6k.ps1 and friends.
#
# A profile is a thin wrapper: it fixes the model path and the llama-server
# flags that belong to that model, and forwards --mode so the backend can still
# be switched at launch.
#
# Usage: ./start-model-template.sh [--mode rocm-cuda] [extra llama-server args]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL="/path/to/your-model.gguf"

MODE="rocm-cuda"
if [[ "${1:-}" == "--mode" ]]; then
    MODE="${2:?}"; shift 2
fi

# --tensor-split on the dual-linux rig:
#   The RTX PRO 6000 has 96 GB and the R9700 has 32 GB — a 3:1 ratio. Device
#   order follows --device, which start-llama-server.sh builds as AMD first,
#   NVIDIA second, so "1,3" matches the hardware. An even "1,1" would cap the
#   model at 64 GB and leave two thirds of the NVIDIA card unused.
#   Retune per model; this is a starting point, not a measured optimum.
exec "$SCRIPT_DIR/start-llama-server.sh" --mode "$MODE" -- \
    -m "$MODEL" \
    -c 8192 \
    -ngl 99 \
    --split-mode layer \
    --tensor-split 1,3 \
    --flash-attn auto \
    "$@"
