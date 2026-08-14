#!/usr/bin/env bash
# DeepSeek V4 Flash MXFP4, both cards + RAM, 128k context.
#
# Thin wrapper: this is what start-deepseek-mxfp4-nvidia-amd-cpu.sh already
# does by default. It exists so the context is visible in the name next to the
# 256k variant, rather than being the one you get by not passing a flag.
#
# 947 pp / 24.6 tg at 16k. 10 expert layers in system RAM.
#
# Usage:
#   ./start-deepseek-mxfp4-nvidia-amd-cpu-128k.sh
#   ./start-deepseek-mxfp4-nvidia-amd-cpu-128k.sh -- --port 8090   # extra llama-server args

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start-deepseek-mxfp4-nvidia-amd-cpu.sh" "$@"
