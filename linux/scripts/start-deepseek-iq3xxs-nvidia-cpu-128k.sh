#!/usr/bin/env bash
# DeepSeek V4 Flash IQ3_XXS, NVIDIA + RAM, 128k context.
#
# Thin wrapper over the profile default. 8 expert layers in system RAM.
# The stable way to run this quant - the all-VRAM dual of the same quant is
# faster at generation but faults (start-deepseek-iq3xxs-nvidia-amd.sh).
#
# Usage:
#   ./start-deepseek-iq3xxs-nvidia-cpu-128k.sh
#   ./start-deepseek-iq3xxs-nvidia-cpu-128k.sh -- --port 8090   # extra llama-server args

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start-deepseek-iq3xxs-nvidia-cpu.sh" "$@"
