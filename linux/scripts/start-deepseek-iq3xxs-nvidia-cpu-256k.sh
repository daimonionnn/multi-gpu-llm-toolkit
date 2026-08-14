#!/usr/bin/env bash
# DeepSeek V4 Flash IQ3_XXS, NVIDIA + RAM, 256k context.
#
# 10 expert layers in system RAM rather than 8, because the bigger KV cache
# takes the VRAM they would have used. Verified with two consecutive
# 261900-token prefills including a full-cache clear.
#
# Usage:
#   ./start-deepseek-iq3xxs-nvidia-cpu-256k.sh
#   ./start-deepseek-iq3xxs-nvidia-cpu-256k.sh -- --port 8090   # extra llama-server args

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start-deepseek-iq3xxs-nvidia-cpu.sh" --256k "$@"
