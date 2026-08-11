#!/usr/bin/env bash
# Step-3.7-Flash Q4_K_S (104 GB, step35 arch, 45 layers, 288 experts / 8 active).
#
# The model dual-GPU exists for: 104 GB does not fit the 96.6 GB NVIDIA card,
# but fits 96.6 + 32.6 GB of combined VRAM with no CPU offload at all. All
# layers, KV and attention stay on CUDA0; the AMD card holds the experts of
# 10 layers (~24 GB) and serves their matmuls in parallel.
#
# Measured: 2100-2440 pp, 78-94 tg, and 384k prefill tokens through the AMD
# expert path with full-cache clears without a fault. The step35 arch is
# unaffected by the deepseek4 HIP bug. Numbers: doc/benchmarks.md.
#
# NOTE: unsloth's UD-Q4_K_XL-R4 (114 GB) will NOT load here - the R4 repack
# uses ggml type 213, an ik_llama.cpp extension outside mainline's range.
#
# Usage:
#   ./start-step37-q4ks-nvidia-amd.sh                  # 128k context (default)
#   ./start-step37-q4ks-nvidia-amd.sh --64k            # 64k, one fewer expert layer on AMD
#   ./start-step37-q4ks-nvidia-amd.sh -- --port 8090   # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

MODEL=/home/matt/.lmstudio/models/stepfun-ai/Step-3.7-Flash-GGUF/Step-3.7-flash-Q4_K_S-00001-of-00003.gguf

# 128k default: the KV cache is bigger on CUDA0, so one more expert layer moves
# to the AMD card to pay for it. Both variants stay entirely in VRAM - nothing
# here is served from system RAM, so the context choice costs capacity, not
# speed, unlike the profiles with a `-cpu` in their name.
CTX=131072
OT='blk\.(3[4-9]|4[0-4])\.ffn_.*_exps.*=ROCm0'
if [[ "${1:-}" == "--64k" ]]; then
    shift
    CTX=65536
    OT='blk\.(3[5-9]|4[0-4])\.ffn_.*_exps.*=ROCm0'
fi
[[ "${1:-}" == "--" ]] && shift

exec "$SCRIPT_DIR/start-llama-server.sh" --mode rocm-cuda \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    -c "$CTX" \
    -ts 0,1 -ot "$OT" \
    -ngl 99 \
    -fa on \
    --jinja \
    "$@"
