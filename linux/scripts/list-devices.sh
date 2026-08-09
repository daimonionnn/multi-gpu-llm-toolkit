#!/usr/bin/env bash
# Enumerate GPUs as llama.cpp sees them, plus the raw vendor view.
# Linux port of list-devices.ps1.
#
# Usage: ./list-devices.sh [--backend rocm-cuda|vulkan|vulkan-cuda]

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BACKEND="rocm-cuda"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── Vendor view — works before anything is built ───────────────────────
info "== NVIDIA =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
               --format=csv,noheader | sed 's/^/  /'
else
    echo "  nvidia-smi not found"
fi

info "== AMD =="
if command -v rocminfo >/dev/null 2>&1; then
    gfx="$(amd_gfx_target || echo unknown)"
    echo "  gfx target: $gfx"
    command -v rocm-smi >/dev/null 2>&1 && \
        rocm-smi --showproductname --showmeminfo vram --csv 2>/dev/null | sed 's/^/  /' | head -6
else
    echo "  rocminfo not found"
fi

info "== Vulkan =="
if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName' | sed 's/^\s*/  /'
else
    echo "  vulkaninfo not found"
fi

# ── llama.cpp view ─────────────────────────────────────────────────────
RUNTIME_DIR="$(runtime_dir_for "$BACKEND")"
SERVER_BIN="$RUNTIME_DIR/llama-server"

echo
if [[ ! -x "$SERVER_BIN" ]]; then
    warn "llama-server not built for backend '$BACKEND' ($RUNTIME_DIR)."
    warn "Run: ./setup-llama.sh --backend $BACKEND"
    exit 0
fi

export_runtime_env "$RUNTIME_DIR"
info "== llama.cpp devices ($BACKEND) =="
echo "  $SERVER_BIN"
"$SERVER_BIN" --list-devices
