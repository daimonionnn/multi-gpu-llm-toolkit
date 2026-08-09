#!/usr/bin/env bash
# Shared helpers for the Linux scripts. Sourced, not executed.
#
# Linux counterpart of the inline helper functions in the PowerShell scripts.
# The two platform-specific problems this file solves:
#
#   1. Picking a CUDA toolkit whose nvcc can actually compile for the installed
#      NVIDIA GPU. A distro-packaged /usr/bin/nvcc often shadows a much newer
#      toolkit in /usr/local/cuda-*, and an nvcc that predates the GPU will fail
#      with "unsupported gpu architecture". This is the direct analogue of
#      Find-CudaCompatibleToolset in setup-llama.ps1.
#
#   2. Locating ROCm, which lives either in /opt/rocm (AMD's own packages) or
#      spread across /usr (distro packages — Ubuntu 26.04 ships it this way,
#      with no /opt/rocm at all).

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_ROOT="$(dirname -- "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname -- "$LINUX_ROOT")"

# ── Output ─────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_INFO=$'\033[36m'; C_OK=$'\033[32m'
    C_WARN=$'\033[33m'; C_ERR=$'\033[31m'
else
    C_RESET=''; C_INFO=''; C_OK=''; C_WARN=''; C_ERR=''
fi

info()  { printf '%s%s%s\n' "$C_INFO" "$*" "$C_RESET"; }
ok()    { printf '%s%s%s\n' "$C_OK"   "$*" "$C_RESET"; }
warn()  { printf '%s%s%s\n' "$C_WARN" "$*" "$C_RESET" >&2; }
die()   { printf '%s%s%s\n' "$C_ERR"  "$*" "$C_RESET" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command '$1' is not in PATH."
}

# ── Runtime directory ──────────────────────────────────────────────────
# Mirrors the runtime-<backend>/ layout used on Windows.
runtime_dir_for() {
    local mode="$1"
    case "$mode" in
        vulkan-cuda)
            if [[ -x "$LINUX_ROOT/runtime-vulkan-cuda/llama-server" ]]; then
                echo "$LINUX_ROOT/runtime-vulkan-cuda"
            else
                echo "$LINUX_ROOT/runtime-vulkan"
            fi ;;
        vulkan*)     echo "$LINUX_ROOT/runtime-vulkan" ;;
        *)           echo "$LINUX_ROOT/runtime-rocm-cuda" ;;
    esac
}

# Backends are built with GGML_BACKEND_DL=ON, so the .so backends are loaded at
# runtime and must be findable. On Windows this was a PATH prepend.
export_runtime_env() {
    local rt="$1"
    export LD_LIBRARY_PATH="$rt${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export GGML_BACKEND_PATH="$rt"
}

# ── NVIDIA ─────────────────────────────────────────────────────────────
# Compute capability of the first NVIDIA GPU, as an integer: "12.0" -> 120.
nvidia_arch() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    [[ -n "$cap" ]] || return 1
    echo "${cap/./}"
}

# Print the path to an nvcc that supports $1 (an arch integer such as 120).
# Prefers the newest toolkit found. Empty output + non-zero status if none.
find_cuda_nvcc() {
    local want="$1" candidates=() n
    [[ -n "${CUDA_HOME:-}"  ]] && candidates+=("$CUDA_HOME/bin/nvcc")
    [[ -n "${CUDA_PATH:-}"  ]] && candidates+=("$CUDA_PATH/bin/nvcc")
    # Newest versioned toolkit first. Depth 3 covers /usr/local/cuda-X.Y/bin/nvcc.
    while IFS= read -r n; do candidates+=("$n"); done < <(
        find /usr/local /opt -maxdepth 3 -path '*cuda*/bin/nvcc' -type f 2>/dev/null | sort -Vr
    )
    command -v nvcc >/dev/null 2>&1 && candidates+=("$(command -v nvcc)")

    for n in "${candidates[@]}"; do
        [[ -x "$n" ]] || continue
        if "$n" --list-gpu-arch 2>/dev/null | grep -qx "compute_$want"; then
            echo "$n"; return 0
        fi
    done
    return 1
}

# ── AMD / ROCm ─────────────────────────────────────────────────────────
# gfx target of the first AMD GPU, e.g. gfx1201.
amd_gfx_target() {
    command -v rocminfo >/dev/null 2>&1 || return 1
    local gfx
    gfx="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v generic | head -1)"
    [[ -n "$gfx" ]] || return 1
    echo "$gfx"
}

# Print "HIPCXX|HIP_PATH|CMAKE_PREFIX_PATH" for the detected ROCm, or fail.
detect_rocm() {
    local root="" hipcxx="" prefix=""

    if command -v hipconfig >/dev/null 2>&1; then
        root="$(hipconfig -R 2>/dev/null || true)"
        local llvm; llvm="$(hipconfig -l 2>/dev/null || true)"
        [[ -x "$llvm/clang++" ]] && hipcxx="$llvm/clang++"
    fi
    if [[ -z "$root" ]]; then
        for r in "${ROCM_PATH:-}" /opt/rocm /usr; do
            [[ -n "$r" && -e "$r/bin/hipcc" ]] && { root="$r"; break; }
        done
    fi
    [[ -n "$root" ]] || return 1

    if [[ -z "$hipcxx" ]]; then
        for c in "$root/llvm/bin/clang++" "$root/bin/amdclang++" "$root/bin/hipcc"; do
            [[ -x "$c" ]] && { hipcxx="$c"; break; }
        done
    fi
    [[ -n "$hipcxx" ]] || return 1

    # CMake package config: AMD layout vs distro multiarch layout.
    for p in "$root/lib/cmake" "/usr/lib/$(uname -m)-linux-gnu/cmake" "$root/lib64/cmake"; do
        [[ -d "$p/hip" ]] && { prefix="$p"; break; }
    done
    [[ -n "$prefix" ]] || return 1

    echo "$hipcxx|$root|$prefix"
}

jobs_count() { nproc 2>/dev/null || echo 4; }
