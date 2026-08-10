#!/usr/bin/env bash
# End-to-end HTTP benchmark against a running llama-server.
# Linux port of benchmark-loaded-model.ps1.
#
# Measures the real client experience: HTTP overhead, prefill, and KV-cache
# context switching under concurrent load. For raw hardware numbers without the
# server in the way, use run-llama-bench.sh instead.
#
# Usage:
#   ./benchmark-loaded-model.sh [--url http://127.0.0.1:8081] \
#       [--contexts 1024,4096,16384] [--predict 256] [--timeout 7200] \
#       [--out-csv results.csv] [--mode rocm-cuda]

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_URL="http://127.0.0.1:8081"
CONTEXTS="1024,4096,16384"
PREDICT_TOKENS=256
TIMEOUT_SEC=7200
OUT_CSV=""
MODE=""
PARALLEL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)      BASE_URL="${2:?}"; shift 2 ;;
        --contexts) CONTEXTS="${2:?}"; shift 2 ;;
        --predict)  PREDICT_TOKENS="${2:?}"; shift 2 ;;
        --timeout)  TIMEOUT_SEC="${2:?}"; shift 2 ;;
        --out-csv)  OUT_CSV="${2:?}"; shift 2 ;;
        --mode)     MODE="${2:?}"; shift 2 ;;
        --parallel) PARALLEL_OVERRIDE="${2:?}"; shift 2 ;;
        -h|--help)  sed -n '2,13p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

require_cmd curl
require_cmd jq

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# ── Health ─────────────────────────────────────────────────────────────
health="$(curl -fsS --max-time 10 "$BASE_URL/health" 2>/dev/null || true)"
[[ "$(jq -r '.status // empty' <<< "$health" 2>/dev/null)" == "ok" ]] || die \
"Cannot reach a healthy llama-server at $BASE_URL. Start your model first, then rerun."

MODEL_NAME="$(curl -fsS --max-time 15 "$BASE_URL/v1/models" 2>/dev/null \
              | jq -r '.data[0].id // "unknown"' 2>/dev/null || echo unknown)"

# ── Pick up mode / args from the running server, for the log ───────────
SERVER_ARGS=""
pid="$(pgrep -x llama-server | head -1 || true)"
if [[ -n "$pid" ]]; then
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    [[ -z "$MODE" && "$exe" =~ runtime-([^/]+)/llama-server ]] && MODE="${BASH_REMATCH[1]}"
    SERVER_ARGS="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
fi

# --parallel N on the server sets how many concurrent requests are worth testing;
# the run sweeps 1..N. An explicit --parallel here overrides the detected value.
MAX_PARALLEL=1
if [[ "$SERVER_ARGS" =~ --parallel[[:space:]]+([0-9]+) ]]; then
    MAX_PARALLEL="${BASH_REMATCH[1]}"
fi
[[ -n "$PARALLEL_OVERRIDE" ]] && MAX_PARALLEL="$PARALLEL_OVERRIDE"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || die "--parallel must be a positive integer."

info "Benchmarking loaded model: $MODEL_NAME"
info "Server:    $BASE_URL"
info "Mode:      ${MODE:-unknown}"
info "Contexts:  $CONTEXTS"
info "Predict:   $PREDICT_TOKENS tokens per run"
info "Parallel:  up to $MAX_PARALLEL concurrent request(s)"
echo

# ── Helpers ────────────────────────────────────────────────────────────
# Token count via the server's own tokenizer. Large texts are tokenized in
# chunks - a single multi-MB /tokenize body can 500 on some servers, and the
# chars/4 fallback once overshot a 262k-context prompt to 283k tokens, which
# is worse than failing. The fallback stays only for small chunks, deflated so
# an estimate can never overshoot the target.
# The chunk goes to jq via --rawfile, never as an argv element: Linux caps a
# single argv string at 128 KiB (MAX_ARG_STRLEN), so `--arg c "$big"` dies with
# E2BIG, the pipe to curl goes empty, and the server logs a confusing
# "attempting to parse an empty input" 500. That failure mode produced a 283k-
# token prompt against a 262k context once; hence also the conservative
# fallback that can only stop prompt growth early, never overshoot it.
token_count() {
    local text="$1" total=0 off=0 resp n
    local CHUNK=262144
    while (( off < ${#text} )); do
        printf '%s' "${text:off:CHUNK}" > "$WORK_DIR/tok_chunk.txt"
        resp="$(jq -nc --rawfile c "$WORK_DIR/tok_chunk.txt" '{content:$c}' \
                | curl -fsS --max-time 120 -H 'Content-Type: application/json' \
                       -d @- "$BASE_URL/tokenize" 2>/dev/null || true)"
        n="$(jq -r 'if type=="array" then length else (.tokens|length) end' <<< "$resp" 2>/dev/null || true)"
        if [[ "$n" =~ ^[0-9]+$ ]]; then
            total=$(( total + n ))
        else
            warn "tokenize failed for a $(wc -c < "$WORK_DIR/tok_chunk.txt")-byte chunk; using a stop-early estimate"
            total=$(( total + CHUNK / 3 ))
        fi
        off=$(( off + CHUNK ))
    done
    echo "$total"
}

# Build a prompt of roughly $1 tokens by repeating a fixed chunk.
build_prompt() {
    local target="$1"
    local chunk="The quick brown fox jumps over the lazy dog near the river bank. "
    local chunk_tokens repeat prompt actual guard=0

    chunk_tokens="$(token_count "$chunk")"
    (( chunk_tokens > 0 )) || chunk_tokens=12

    repeat=$(( (target + chunk_tokens - 1) / chunk_tokens ))
    (( repeat > 0 )) || repeat=1

    prompt="$(printf "%.0s$chunk" $(seq 1 "$repeat"))"
    actual="$(token_count "$prompt")"

    while (( actual < target && guard < 4 )); do
        local missing=$(( target - actual ))
        local more=$(( (missing + chunk_tokens - 1) / chunk_tokens ))
        (( more > 0 )) || more=1
        prompt+="$(printf "%.0s$chunk" $(seq 1 "$more"))"
        actual="$(token_count "$prompt")"
        (( guard++ ))
    done

    printf '%s' "$prompt" > "$WORK_DIR/prompt.txt"
    echo "$actual"
}

# ── Run ────────────────────────────────────────────────────────────────
RESULTS="$WORK_DIR/results.csv"
echo "Context,ParallelReqs,PromptTokens,PredictTokens,PrefillTokPerSec,GenTokPerSec,TotalTokPerSec,TotalWallMs,Status" > "$RESULTS"

IFS=',' read -ra CTX_LIST <<< "$CONTEXTS"
for ctx in "${CTX_LIST[@]}"; do
    ctx="${ctx// /}"
    target=$(( ctx - PREDICT_TOKENS - 32 ))
    (( target < 256 )) && target=256

    warn "Preparing prompt for context $ctx (target ~$target prompt tokens) ..."
    prompt_tokens="$(build_prompt "$target")"

    # ignore_eos is required, not cosmetic: on a synthetic prompt the model
    # usually emits EOS on the first token, and the run then reports a
    # generation rate computed from a single token.
    jq -nc --rawfile p "$WORK_DIR/prompt.txt" --argjson n "$PREDICT_TOKENS" \
       '{prompt:$p, n_predict:$n, temperature:0.1, top_p:0.95, top_k:40,
         cache_prompt:false, stream:false, ignore_eos:true}' > "$WORK_DIR/payload.json"

    for (( p = 1; p <= MAX_PARALLEL; p++ )); do
        warn "Context $ctx — prompt tokens $prompt_tokens — parallel requests: $p"
        rm -f "$WORK_DIR"/resp_*.json

        start_ns="$(date +%s%N)"
        for (( k = 1; k <= p; k++ )); do
            curl -fsS --max-time "$TIMEOUT_SEC" \
                 -H 'Content-Type: application/json' \
                 -d @"$WORK_DIR/payload.json" \
                 "$BASE_URL/completion" > "$WORK_DIR/resp_$k.json" 2>/dev/null &
        done
        wait
        end_ns="$(date +%s%N)"
        wall_ms=$(( (end_ns - start_ns) / 1000000 ))

        # Aggregate the per-request timings blocks.
        read -r successes gen_tok gen_tps pre_tok pre_tps < <(
            cat "$WORK_DIR"/resp_*.json 2>/dev/null \
            | jq -s 'map(select(.timings != null))
                     | [ length,
                         (map(.timings.predicted_n)          | add // 0),
                         (map(.timings.predicted_per_second) | add // 0),
                         (map(.timings.prompt_n)             | add // 0),
                         (map(.timings.prompt_per_second)    | add // 0) ]
                     | @tsv' -r 2>/dev/null || echo "0 0 0 0 0"
        )
        successes="${successes:-0}"

        if (( successes > 0 )); then
            status="ok"
            (( successes == p )) || status="partial: $successes/$p succeeded"
            read -r avg_pre_tok avg_gen_tok avg_pre_tps avg_gen_tps tot_tps < <(
                jq -nr --argjson s "$successes"   --argjson gt "$gen_tok" \
                       --argjson gp "$gen_tps"    --argjson pt "$pre_tok" \
                       --argjson pp "$pre_tps"    --argjson w  "$wall_ms" \
                   '[ ($pt/$s|round), ($gt/$s|round),
                      ($pp/$s*100|round/100), ($gp/$s*100|round/100),
                      (if $w > 0 then ($gt/($w/1000)*100|round/100) else 0 end) ] | @tsv'
            )
        else
            status="failed: no successful responses"
            avg_pre_tok="$prompt_tokens"; avg_gen_tok=0
            avg_pre_tps=0; avg_gen_tps=0; tot_tps=0
        fi

        echo "$ctx,$p,$avg_pre_tok,$avg_gen_tok,$avg_pre_tps,$avg_gen_tps,$tot_tps,$wall_ms,$status" >> "$RESULTS"
    done
    echo
done

# ── Report ─────────────────────────────────────────────────────────────
ok "Benchmark results:"
column -t -s, "$RESULTS"

if [[ -n "$OUT_CSV" ]]; then
    cp "$RESULTS" "$OUT_CSV"
    ok "CSV written to: $OUT_CSV"
fi

LOG_PATH="$SCRIPT_DIR/benchmark.log"
{
    echo "=========================================="
    echo "Date:      $(date -Is)"
    echo "Rig:       $(uname -n) ($(uname -r))"
    echo "Model:     $MODEL_NAME"
    echo "Mode:      ${MODE:-unknown}"
    echo "ServerCmd: $SERVER_ARGS"
    echo "Contexts:  $CONTEXTS"
    echo "Predict:   $PREDICT_TOKENS"
    echo
    column -t -s, "$RESULTS"
    echo
} >> "$LOG_PATH"
ok "Benchmark log appended to: $LOG_PATH"
