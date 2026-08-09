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
BATCH_SIZES=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?}"; shift 2 ;;
        -p)     PROMPT_TOKENS="${2:?}"; shift 2 ;;
        -n)     PREDICT_TOKENS="${2:?}"; shift 2 ;;
        -b)     BATCH_SIZES="${2:?}"; shift 2 ;;   # llama-bench batch size, NOT parallel requests
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        --)     shift; EXTRA_ARGS=("$@"); break ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── Auto-detect from a running server ──────────────────────────────────
# llama-bench understands only a small subset of llama-server's flags, and two
# of the shared ones take a different separator. Rather than denylisting the
# server-only flags (there are dozens, plus fork-specific ones), allowlist what
# llama-bench actually accepts and translate where the syntax differs.
BENCH_FLAGS_WITH_VALUE=(
    -m --model -ngl --n-gpu-layers -sm --split-mode -mg --main-gpu
    -ctk --cache-type-k -ctv --cache-type-v -fa --flash-attn
    -ub --ubatch-size -nkvo --no-kv-offload -ot --override-tensor
    -ncmoe --n-cpu-moe --numa
)
# These mean the same thing but llama-server separates with "," and
# llama-bench with "/" — passing the server's form through is a hard error.
BENCH_FLAGS_SLASH_SEPARATED=(-dev --device -ts --tensor-split)

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

    dropped=()
    take_next=""      # flag name whose value we still need
    translate=0
    for a in "${raw_args[@]}"; do
        if [[ -n "$take_next" ]]; then
            if (( translate )); then
                EXTRA_ARGS+=("$take_next" "${a//,//}")
            else
                EXTRA_ARGS+=("$take_next" "$a")
            fi
            take_next=""; translate=0; continue
        fi
        keep=0
        for f in "${BENCH_FLAGS_WITH_VALUE[@]}"; do
            [[ "$a" == "$f" ]] && { take_next="$a"; keep=1; break; }
        done
        (( keep )) && continue
        for f in "${BENCH_FLAGS_SLASH_SEPARATED[@]}"; do
            [[ "$a" == "$f" ]] && { take_next="$a"; translate=1; keep=1; break; }
        done
        (( keep )) && continue
        [[ "$a" == -* ]] && dropped+=("$a")
    done

    if (( ${#dropped[@]} > 0 )); then
        warn "Dropped server-only flags llama-bench does not accept: ${dropped[*]}"
    fi
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
info "  Prompts:   $PROMPT_TOKENS   Predict: $PREDICT_TOKENS   Batch: ${BATCH_SIZES:-default}"
info "======================================"
echo

bench_cmd=("$BENCH_BIN" -p "$PROMPT_TOKENS" -n "$PREDICT_TOKENS")
# Only pass -b when asked. In llama-bench -b is the logical batch size (default
# 2048), not a count of parallel requests — forcing it to a small value makes
# prompt processing pathologically slow rather than measuring concurrency.
[[ -n "$BATCH_SIZES" ]] && bench_cmd+=(-b "$BATCH_SIZES")
bench_cmd+=("${EXTRA_ARGS[@]}")

if "${bench_cmd[@]}"; then
    ok "llama-bench finished successfully."
else
    die "llama-bench exited with code $?"
fi
