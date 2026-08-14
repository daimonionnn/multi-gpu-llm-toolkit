#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 MXFP4 (146 GB) - the lossless reference quant.
#
# The model is QAT with native MXFP4 experts, so this quant IS the original
# weights; every other quant is equal at best (Q8) or lossy (Q4 and below).
# Too large for VRAM alone: expert-offload dual plus experts of the first
# 10-12 layers in system RAM.
#
# STABILITY: the dual layouts here are NOT safe for unattended duty. This
# header used to claim MXFP4 was exempt from the HIP fault because it is not
# an IQ quant; that was disproved in production on 2026-08-09, when the MXFP4
# dual faulted at ~45k prefill tokens hours after passing its gauntlet. The
# fault is low-rate and probabilistic on the AMD expert path, so passing a
# gauntlet certifies nothing. Use --cuda-only for anything that must stay up.
#
# Sibling profiles: start-deepseek-iq2xxs-nvidia.sh (single-card fits),
# start-deepseek-q2kxl-nvidia-amd.sh (all-VRAM dual). Numbers: doc/benchmarks.md.
#
# Usage:
#   ./start-deepseek-mxfp4-nvidia-amd-cpu.sh                 # 128k context (default)
#   ./start-deepseek-mxfp4-nvidia-amd-cpu.sh --256k          # full 262144 context, ~20% slower prefill
#   ./start-deepseek-mxfp4-nvidia-amd-cpu.sh --cuda-only     # no AMD card: slower (pp ~305, tg ~16.4)
#                                             # but immune to the ROCm faults - use
#                                             # for unattended/fallback service duty
#   ./start-deepseek-mxfp4-nvidia-amd-cpu.sh -- --port 8090  # extra llama-server args

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"

MODEL=/home/matt/.lmstudio/models/lmstudio-community/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf

# Default: 128k. The smaller KV cache frees VRAM that then holds two more
# expert layers, so only 10 go to system RAM instead of 12 - and DDR5 at
# ~75 GB/s is the bottleneck, not the GPUs. That makes 128k *faster*, not just
# smaller: 480-592 pp against 386 at 256k, gauntlet-verified, 21-25 tg.
#
# 256k was the default until 2026-08-10. It was the right default while this
# was the "lossless reference at full context" profile; it is the wrong one now
# that the profile's main job is answering hermes, where prefill latency before
# the first token is what the user feels. A 63.5k-token conversation prefills
# in ~115 s here against ~165 s at 256k.
MODE_ARGS=(--mode rocm-cuda)
OT_ARGS=(-ts 0,1 -ot 'blk\.(3[5-9]|4[0-2])\.ffn_.*_exps.*=ROCm0')
MODEL_ARGS=(-c 131072 --n-cpu-moe 10)
if [[ "${1:-}" == "--256k" ]]; then
    shift
    # KV needs ~13 GB on CUDA0, so two more expert layers are exiled to RAM.
    # Verified: 386 pp / 18.4 tg on a full 261900-token prompt, incl. the
    # clear-and-refill cycle.
    MODEL_ARGS=(-c 262144 --n-cpu-moe 12)
elif [[ "${1:-}" == "--cuda-only" ]]; then
    shift
    # The AMD expert path faults intermittently on ALL quants (reproduced on
    # MXFP4 in production at ~45k prefill tokens, 2026-08-09, despite passing
    # the gauntlets - low-rate, probabilistic). CUDA-only never faulted once.
    MODE_ARGS=(--mode cuda)
    OT_ARGS=()
    MODEL_ARGS=(-c 131072 --n-cpu-moe 18)
fi
[[ "${1:-}" == "--" ]] && shift

# -b 4096 -ub 2048 is worth +55-60% of prefill over llama.cpp's 2048/512
# (937/947/907 against 565/595/597 pp at 4k/16k/32k), at no cost to
# generation. Larger micro-batches let more tokens share one load of an
# expert's weights, which is the dominant cost once experts live off the GPU.
# 2048 and not more: -ub 4096 loads and serves 4k, then cannot allocate at 16k.
# Numbers in doc/benchmarks.md.
#
# --no-mmap is unconditional here because every mode offloads experts to system
# RAM (--n-cpu-moe 10/12/18), and llama.cpp warns on startup that mmap costs
# performance whenever tensors are overridden to the CPU. The all-VRAM profiles
# (start-deepseek-q2kxl-nvidia-amd.sh, start-step37-q4ks-nvidia-amd.sh,
# start-gptoss-mxfp4-nvidia.sh) deliberately keep mmap: with nothing in host RAM there is nothing to speed up, and the
# flag would only slow the load and give up page-cache sharing.
exec "$SCRIPT_DIR/start-llama-server.sh" "${MODE_ARGS[@]}" \
    --runtime "$LINUX_ROOT/runtime-rocm-cuda128" -- \
    -m "$MODEL" \
    "${MODEL_ARGS[@]}" \
    ${OT_ARGS[@]+"${OT_ARGS[@]}"} \
    -ngl 99 \
    -fa on \
    -b 4096 -ub 2048 \
    --no-mmap \
    --jinja \
    "$@"
