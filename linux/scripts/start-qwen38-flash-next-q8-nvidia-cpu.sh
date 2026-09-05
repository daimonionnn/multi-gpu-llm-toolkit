#!/usr/bin/env bash
# Qwen3.8-Flash-Next Q8_0 (175 GiB) - CUDA-only, experts in system RAM, 128k ctx.
#
# The largest model this rig runs, and the fastest of the large ones: 2008 pp /
# 21.9 tg on a full 128k prompt, against DeepSeek V4 Flash MXFP4's 1180 / 16.7
# in the same placement. The file is bigger (175 GiB against 146) and it still
# wins, because the architecture activates far less of itself per token -
# 10 experts of 512, each only 640 wide, plus hybrid SSM layers that carry
# state instead of a growing KV cache.
#
#   4k    2599 pp / 28.7 tg    1.5 s to first token
#   32k   3003 pp / 26.3 tg   10.9 s
#   65k   2661 pp / 24.0 tg   24.5 s
#   128k  2008 pp / 21.9 tg   65.1 s
#
# Vision is on by default: the repo ships an mmproj beside the weights and this
# model reads images through it. Costs ~1.1 GB of VRAM, which is why the headroom
# note below matters; --no-vision gives it back.
#
# Usage:
#   ./start-qwen38-flash-next-q8-nvidia-cpu.sh                # 128k, vision on
#   ./start-qwen38-flash-next-q8-nvidia-cpu.sh --no-vision    # text only
#   ./start-qwen38-flash-next-q8-nvidia-cpu.sh -- --port 8090 # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

# GGUF root; override with LLM_MODELS_DIR. Default is LM Studio's download dir.
MODELS_DIR="${LLM_MODELS_DIR:-$HOME/.lmstudio/models}"

MODEL="$MODELS_DIR/lmstudio-community/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"
[[ -f "$MODEL" ]] || {
    echo "Model not found: $MODEL" >&2
    echo "Download it first, e.g. in LM Studio search 'lmstudio-community Qwen3.8-Flash-Next Q8_0'" >&2
    echo 'Or set LLM_MODELS_DIR if your GGUFs live outside $HOME/.lmstudio/models.' >&2
    exit 1
}

# The projector is a separate GGUF next to the weights (qwen3vl_merger, 0.85 GiB
# on disk, ~1.1 GB resident). Missing it is not fatal - the model still serves
# text - so this warns rather than exiting.
MMPROJ="$MODELS_DIR/lmstudio-community/Qwen3.8-Flash-Next-GGUF/mmproj-Qwen3.8-Flash-Next-BF16.gguf"
VISION_ARGS=()
if [[ "${1:-}" == "--no-vision" ]]; then
    shift
elif [[ -f "$MMPROJ" ]]; then
    VISION_ARGS=(--mmproj "$MMPROJ")
else
    echo "Projector not found, serving text only: $MMPROJ" >&2
fi

# NOT runtime-rocm-cuda128, unlike every other profile here. This model's
# architecture is `qwen4exp`, which upstream added after that runtime was built,
# so it is the one profile that needs the newer engine (b10428-392-g74a7c897f).
#
# runtime-cuda128 is CUDA-only and internally consistent - server, libllama and
# the CUDA backend all come from the same commit. The dual runtime deliberately
# pairs a locally built HIP backend with a container-built CUDA one, so pointing
# this at it would mix engine versions. Nothing here needs the AMD card anyway.
RUNTIME="$LINUX_ROOT/runtime-cuda128"
[[ -x "$RUNTIME/llama-server" ]] || {
    echo "Runtime not found: $RUNTIME" >&2
    echo "Build it with ./build-cuda12-container.sh (needs docker)." >&2
    exit 1
}

# --n-cpu-moe 18 is the floor that fits, and lower is better all the way down to
# it - measured at 4k/32k, monotonic, no knee:
#
#   ncmoe   VRAM       pp 4k    pp 32k   tg 32k
#   24      79.3 GB    2186     2570     21.2
#   22      84.3 GB    2320     2690     22.8
#   20      89.4 GB    2438     2848     24.5
#   18      94.5 GB    2599     3003     26.3   <- shipped
#
# 16 would need ~99.6 GB and does not fit. 18 leaves ~3.3 GB of headroom, and the
# projector spends ~1.1 GB of that: measured 95.6 GB resident with vision on, and
# 95.8 GB while encoding a 1024x1024 image. That is enough only because this card
# drives no display. With a desktop on the same GPU, or for a wide margin on
# large images, use 20 - it costs ~5% of prefill and buys 5 GB.
#
# -b 8192 -ub 4096 and threads from nproc are carried over from the DeepSeek
# profile rather than re-swept for this model - see doc/benchmarks.md for why
# those are the right shape, and treat them as a starting point, not a measured
# optimum for this architecture.
[[ "${1:-}" == "--" ]] && shift

exec "$SCRIPT_DIR/start-llama-server.sh" --mode cuda \
    --runtime "$RUNTIME" -- \
    -m "$MODEL" \
    ${VISION_ARGS[@]+"${VISION_ARGS[@]}"} \
    -c 131072 \
    --n-cpu-moe 18 \
    -ngl 99 \
    -fa on \
    -b 8192 -ub 4096 \
    -t "$(nproc)" -tb "$(nproc)" \
    --no-mmap \
    --jinja \
    "$@"
