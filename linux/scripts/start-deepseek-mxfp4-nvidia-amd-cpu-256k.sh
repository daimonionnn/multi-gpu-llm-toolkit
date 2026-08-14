#!/usr/bin/env bash
# DeepSeek V4 Flash MXFP4, both cards + RAM, full 256k context.
#
# 839 pp / 21.4 tg at 16k, against 947/24.6 for the 128k profile: the larger
# KV cache takes VRAM that would otherwise hold expert weights, so two more
# layers (12 rather than 10) are served from system RAM. Pay that only if you
# need the window.
#
# Usage:
#   ./start-deepseek-mxfp4-nvidia-amd-cpu-256k.sh
#   ./start-deepseek-mxfp4-nvidia-amd-cpu-256k.sh -- --port 8090   # extra llama-server args

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start-deepseek-mxfp4-nvidia-amd-cpu.sh" --256k "$@"
