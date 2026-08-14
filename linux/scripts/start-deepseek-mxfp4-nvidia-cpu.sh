#!/usr/bin/env bash
# DeepSeek V4 Flash MXFP4 on the NVIDIA card only - no AMD involvement.
#
# Thin wrapper over start-deepseek-mxfp4-nvidia-amd-cpu.sh --cuda-only. Immune
# to the probabilistic ROCm expert fault, so this is the profile for
# unattended/fallback service duty; the systemd unit deepseek-server.service
# runs this configuration.
#
# It is no longer the slow option: with -b 4096 -ub 2048 it measures 1175 pp /
# 17.1 tg at 16k, against ~305 pp before that flag. It keeps 18 expert layers
# in system RAM, so more of its work is weight transfer than the dual's, and
# more of it is amortised by the larger micro-batch.
#
# PORTS: running this by hand binds 8081, the launcher default. The hermes
# fallback looks for 8099, which is the port the systemd unit passes in - so
# starting it by hand does NOT give hermes a fallback, it gives you a second
# server somewhere hermes is not looking. For fallback duty always use the
# service; it also comes back after a reboot.
#
#   systemctl --user start deepseek-server     # 8099, what hermes expects
#   systemctl --user status deepseek-server
#
# Usage (interactive, port 8081):
#   ./start-deepseek-mxfp4-nvidia-cpu.sh
#   ./start-deepseek-mxfp4-nvidia-cpu.sh -- --port 8099   # extra llama-server args

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start-deepseek-mxfp4-nvidia-amd-cpu.sh" --cuda-only "$@"
