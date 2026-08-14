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
# PORT: 8099 whichever way it is started. This profile IS the hermes fallback,
# and hermes only looks at 8099, so binding the launcher's usual 8081 by hand
# produced a server that looked perfectly healthy while hermes still reported
# the fallback unreachable. There is no reason for this one to live anywhere
# else, so the port is pinned here rather than left to the launcher default.
#
# A consequence worth knowing: if the systemd unit is already running, starting
# this by hand now fails to bind instead of silently creating a second, useless
# server. That is the intended behaviour - check with
# `systemctl --user status deepseek-server` first.
#
# Usage:
#   ./start-deepseek-mxfp4-nvidia-cpu.sh                  # port 8099
#   ./start-deepseek-mxfp4-nvidia-cpu.sh -- --port 9000   # override; last wins
#   systemctl --user start deepseek-server                # the same, as a service

# A user-supplied --port lands after ours, and llama.cpp takes the last one.
[[ "${1:-}" == "--" ]] && shift
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start-deepseek-mxfp4-nvidia-amd-cpu.sh" \
    --cuda-only -- --port 8099 "$@"
