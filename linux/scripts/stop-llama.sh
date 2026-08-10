#!/usr/bin/env bash
# Stop whatever this project is serving and hand the GPUs back.
#
# The recurring failure this exists for: a server is already resident, so the
# next launch either OOMs, or reports "no CUDA device detected" because the
# previous 146 GB model is still tearing down, or dies on a port that
# something else owns. All three look like different bugs and are the same
# situation - something is still holding a resource.
#
# Usage:
#   ./stop-llama.sh              stop this project's servers, wait for the VRAM
#   ./stop-llama.sh --status     report only, change nothing
#   ./stop-llama.sh --all        also ask LM Studio / Ollama to unload
#
# Notes:
#   - Stopping the systemd unit is not optional. deepseek-server has
#     Restart=on-failure, and a killed process exits 143, which systemd counts
#     as a failure and restarts 15 seconds later.
#   - Ports are reported, never killed. Today's "couldn't bind" was an
#     unrelated Caddy container on 8080; killing a stranger's process to free
#     a port is not this script's business.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# common.sh sets `set -euo pipefail`, which is wrong for a cleanup script: a
# grep that finds nothing is the normal case here (a free port, no matching
# unit), and under -e the first one aborts the script mid-way, leaving
# processes running and printing no report - exactly the silent half-finished
# state this exists to prevent. Keep -u, drop the other two.
set +e +o pipefail

STATUS_ONLY=0
EVICT_OTHERS=0
PORTS=(8080 8081 8090 8099)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)  STATUS_ONLY=1; shift ;;
        --all)     EVICT_OTHERS=1; shift ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── Reporting ──────────────────────────────────────────────────────────
gpu_report() {
    local used total
    if command -v nvidia-smi >/dev/null; then
        read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
        [[ -n "${used:-}" ]] && printf '  NVIDIA  %6s / %6s MiB\n' "$used" "$total"
        nvidia-smi --query-compute-apps=pid,used_memory,process_name \
            --format=csv,noheader 2>/dev/null \
            | sed 's/^/          /'
    fi
    if command -v rocm-smi >/dev/null; then
        local b
        b="$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used Memory' | grep -oE '[0-9]+$' | head -1)"
        [[ -n "$b" ]] && printf '  AMD     %6d /  32620 MiB\n' "$(( b / 1048576 ))"
        rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+\t/ {printf "          %s  %s\n", $1, $2}'
    fi
}

port_report() {
    local p line
    for p in "${PORTS[@]}"; do
        line="$(ss -tlnp 2>/dev/null | grep ":$p " | head -1)"
        [[ -z "$line" ]] && continue
        if [[ "$line" =~ users:\(\(\"([^\"]+)\",pid=([0-9]+) ]]; then
            printf '  %-5s held by %s (pid %s)\n' "$p" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        else
            # ss only names processes you own; anything else needs root
            printf '  %-5s held by another user'"'"'s process (try: sudo ss -tlnp | grep :%s)\n' "$p" "$p"
        fi
    done
}

info "Before:"
gpu_report
port_report

if (( STATUS_ONLY )); then
    exit 0
fi

# ── Stop our systemd units ─────────────────────────────────────────────
# Found by what they run, not by name, so a second profile added later as a
# service is picked up without editing this script.
units="$(systemctl --user list-units --type=service --all --no-legend 2>/dev/null \
         | awk '{print $1}' \
         | while read -r u; do
               systemctl --user show "$u" -p ExecStart --value 2>/dev/null \
                   | grep -q "$SCRIPT_DIR" && echo "$u"
           done)"
for u in $units; do
    if [[ "$(systemctl --user is-active "$u" 2>/dev/null)" == "active" ]]; then
        info "Stopping $u ..."
        systemctl --user stop "$u"
    fi
done

# ── Stop llama-server ──────────────────────────────────────────────────
# By PID, never `pkill -f llama-server`: that pattern also matches the shell
# running this script when it was launched from a command line containing the
# word, and the script kills itself instead (exit 144).
pids="$(pgrep -x llama-server 2>/dev/null || true)"
if [[ -n "$pids" ]]; then
    info "Stopping llama-server: $(echo "$pids" | tr '\n' ' ')"
    for p in $pids; do kill "$p" 2>/dev/null; done
    for _ in $(seq 1 30); do pgrep -x llama-server >/dev/null || break; sleep 1; done
    pids="$(pgrep -x llama-server 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        warn "Still alive after 30s, sending SIGKILL"
        for p in $pids; do kill -9 "$p" 2>/dev/null; done
    fi
fi

# ── Optionally ask the other model hosts to unload ──────────────────────
if (( EVICT_OTHERS )); then
    if command -v lms >/dev/null; then
        info "Asking LM Studio to unload ..."; lms unload --all >/dev/null 2>&1 || true
    fi
    if command -v ollama >/dev/null; then
        ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | while read -r m; do
            [[ -n "$m" ]] && { info "Stopping ollama model $m ..."; ollama stop "$m" >/dev/null 2>&1 || true; }
        done
    fi
fi

# ── Wait for the VRAM to actually come back ─────────────────────────────
# The point of the whole script. A 146 GB model does not release instantly,
# and a launch during teardown fails with "no CUDA device detected", which
# reads like a driver problem rather than a timing one.
if command -v nvidia-smi >/dev/null; then
    for _ in $(seq 1 150); do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
        [[ -n "$used" && "$used" -lt 4000 ]] && break
        sleep 2
    done
    [[ -n "${used:-}" && "$used" -ge 4000 ]] && \
        warn "NVIDIA still holds ${used} MiB after 300s - see the process list below"
fi

echo
info "After:"
gpu_report
port_report

remaining="$(pgrep -cx llama-server 2>/dev/null || true)"
if [[ "${remaining:-0}" -gt 0 ]]; then
    die "$remaining llama-server process(es) still running"
fi
ok "GPUs released."
