#!/usr/bin/env bash
# DeepSeek V4 Flash MXFP4 on the NVIDIA card only - no AMD involvement.
#
# Thin wrapper over start-deepseek-mxfp4.sh --cuda-only: slower than the dual
# (pp ~305 / tg ~16.4 vs ~390/23) but immune to the probabilistic ROCm expert
# fault, so this is the profile for unattended/fallback service duty. The
# systemd unit deepseek-server.service runs this configuration.
#
# Usage:
#   ./start-deepseek-mxfp4-cuda-only.sh
#   ./start-deepseek-mxfp4-cuda-only.sh -- --port 8099   # extra llama-server args

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start-deepseek-mxfp4.sh" --cuda-only "$@"
