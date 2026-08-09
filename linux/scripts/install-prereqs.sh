#!/usr/bin/env bash
# Install build prerequisites. Linux port of install-prereqs.ps1 (winget -> apt).
#
# Debian/Ubuntu only. On other distros use it as a package checklist.
# Does NOT install GPU drivers — those should already be working; verify with
# ./list-devices.sh first.
#
# Usage: ./install-prereqs.sh [--backend rocm-cuda|vulkan|vulkan-cuda|all] [--dry-run]

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BACKEND="all"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="${2:?}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

command -v apt-get >/dev/null 2>&1 || die \
"This script only handles Debian/Ubuntu. Install the equivalents for your distro:
  build tools: gcc, g++, cmake, ninja, git
  ROCm:        hipcc + hipblas/rocblas development packages
  Vulkan:      glslc, Vulkan loader and headers
  CUDA:        a CUDA Toolkit new enough for your GPU"

pkgs=(build-essential cmake ninja-build git pkg-config curl jq libcurl4-openssl-dev)

case "$BACKEND" in
    rocm-cuda)   pkgs+=(hipcc libhipblas-dev librocblas-dev rocminfo rocm-smi) ;;
    vulkan)      pkgs+=(glslc libvulkan-dev vulkan-tools) ;;
    vulkan-cuda) pkgs+=(glslc libvulkan-dev vulkan-tools) ;;
    all)         pkgs+=(hipcc libhipblas-dev librocblas-dev rocminfo rocm-smi
                        glslc libvulkan-dev vulkan-tools) ;;
    *) die "Invalid backend '$BACKEND'." ;;
esac

info "Packages to install:"
printf '  %s\n' "${pkgs[@]}"
echo

if (( DRY_RUN )); then
    warn "--dry-run: nothing installed."
    exit 0
fi

sudo apt-get update
sudo apt-get install -y "${pkgs[@]}"

echo
# The CUDA Toolkit is deliberately not in the list: the version needed depends
# on the GPU, and distro packages are frequently too old for a recent card.
if [[ "$BACKEND" == "rocm-cuda" || "$BACKEND" == "vulkan-cuda" || "$BACKEND" == "all" ]]; then
    arch="$(nvidia_arch 2>/dev/null || true)"
    if [[ -n "$arch" ]]; then
        if find_cuda_nvcc "$arch" >/dev/null 2>&1; then
            ok "CUDA toolkit supporting sm_$arch found: $(find_cuda_nvcc "$arch")"
        else
            warn "No CUDA toolkit found that supports your GPU (sm_$arch)."
            warn "Install a recent CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
            warn "A distro-packaged nvcc is often too old for a current-generation card."
        fi
    fi
fi

ok "Done. Next: ./list-devices.sh, then ./setup-llama.sh --backend <backend>"
