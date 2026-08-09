#!/usr/bin/env bash
# Raw hardware benchmark via llama-bench (no HTTP, no server queue).
# Linux port of run-llama-bench.ps1.
#
# With no arguments it reads the parameters off a running llama-server, the same
# way the PowerShell version does — /proc instead of WMI.
#
# Usage:
#   ./run-llama-bench.sh [--mode rocm-cuda] [-p 128,512] [-n 128,256] [-b 1,2,4]
#                        [-- <extra llama-bench args>]

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE=""
PROMPT_TOKENS="128,512,1024"
PREDICT_TOKENS="128,256"
BATCH_SIZES="1,2,4"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?}"; shift 2 ;;
        -p)     PROMPT_TOKENS="${2:?}"; shift 2 ;;
        -n)     PREDICT_TOKENS="${2:?}"; shift 2 ;;
        -b)     BATCH_SIZES="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        --)     shift; EXTRA_ARGS=("$@"); break ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── Auto-detect from a running server ──────────────────────────────────
# llama-bench does not accept these; drop them and their values.
SERVER_ONLY_WITH_VALUE=(--host --port --parallel --api-key --path --timeout)
SERVER_ONLY_FLAGS=(--webui --device)

if [[ -z "$MODE" && ${#EXTRA_ARGS[@]} -eq 0 ]]; then
    pid="$(pgrep -x llama-server | head -1 || true)"
    [[ -n "$pid" ]] || die \
"llama-server is not running and no arguments were provided.
Either start your server first to auto-detect arguments, or pass --mode and -- <args> manually."

    warn "llama-server is currently running (pid $pid)."
    warn "Running llama-bench against the same model concurrently may cause OOM — consider stopping the server first."

    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    [[ "$exe" =~ runtime-([^/]+)/llama-server ]] && MODE="${BASH_REMATCH[1]}"

    mapfile -d '' -t raw_args < "/proc/$pid/cmdline"
    unset 'raw_args[0]'  # the executable itself

    skip_next=0
    for a in "${raw_args[@]}"; do
        if (( skip_next )); then skip_next=0; continue; fi
        drop=0
        for s in "${SERVER_ONLY_WITH_VALUE[@]}"; do
            [[ "$a" == "$s" ]] && { drop=1; skip_next=1; break; }
        done
        (( drop )) && continue
        for s in "${SERVER_ONLY_FLAGS[@]}"; do
            [[ "$a" == "$s" ]] && { drop=1; [[ "$a" == "--device" ]] && skip_next=1; break; }
        done
        (( drop )) && continue
        EXTRA_ARGS+=("$a")
    done
fi

[[ -n "$MODE" ]] || die "Could not determine the runtime mode. Pass --mode explicitly."

BENCH_BIN="$LINUX_ROOT/runtime-$MODE/llama-bench"
[[ -x "$BENCH_BIN" ]] || die \
"Cannot find llama-bench at $BENCH_BIN
It was not built for the '$MODE' runtime. Re-run:
  ./setup-llama.sh --backend $MODE"

export_runtime_env "$LINUX_ROOT/runtime-$MODE"

info "======================================"
info "llama-bench hardware test"
info "  Mode:      $MODE"
info "  Base args: ${EXTRA_ARGS[*]}"
info "  Prompts:   $PROMPT_TOKENS   Predict: $PREDICT_TOKENS   Batch: $BATCH_SIZES"
info "======================================"
echo

if "$BENCH_BIN" -p "$PROMPT_TOKENS" -n "$PREDICT_TOKENS" -b "$BATCH_SIZES" "${EXTRA_ARGS[@]}"; then
    ok "llama-bench finished successfully."
else
    die "llama-bench exited with code $?"
fi
