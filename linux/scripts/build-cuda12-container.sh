#!/usr/bin/env bash
# Build the CUDA backend with CUDA 12.8 inside a container, then assemble a
# dual-vendor runtime that combines it with the locally built HIP backend.
#
# Why this exists: CUDA 13.x-built flash-attention is several times slower on
# Blackwell (sm_120) past 8192 context, and CUDA 12.x cannot be compiled on a
# glibc 2.43 host directly. See doc/cuda-fa-blackwell.md. NVIDIA's migration
# guide recommends CUDA 12.8 for sm_120.
#
# Produces:
#   runtime-cuda128/       CUDA-only runtime (server, bench, bundled CUDA 12 libs)
#   runtime-rocm-cuda128/  dual runtime, if runtime-rocm-cuda exists to merge with
#
# Usage: ./build-cuda12-container.sh [--image IMG] [--repo DIR]

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

IMAGE="nvidia/cuda:12.8.0-devel-ubuntu22.04"
REPO_DIR="$LINUX_ROOT/llama.cpp"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="${2:?}"; shift 2 ;;
        --repo)  REPO_DIR="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

require_cmd docker
require_cmd git
[[ -d "$REPO_DIR/.git" ]] || die "llama.cpp checkout not found at $REPO_DIR. Run setup-llama.sh first."

arch="$(nvidia_arch)" || die "nvidia-smi found no GPU."
CUDA_ARCH="${arch}a-real"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/src" "$WORK/out"

# Committed tree only: local patches for the 13.x builds must not leak in here.
info "Exporting stock source at $(git -C "$REPO_DIR" rev-parse --short HEAD) ..."
git -C "$REPO_DIR" archive HEAD | tar -x -C "$WORK/src"

info "Building CUDA backend in $IMAGE (arch $CUDA_ARCH) ..."
docker run --rm -v "$WORK/src":/src:ro -v "$WORK/out":/out "$IMAGE" bash -c '
set -e
chmod 1777 /tmp
apt-get update -qq >/dev/null 2>&1 || apt-get update
apt-get install -y -qq cmake ninja-build git > /dev/null
cmake -S /src -B /b -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
  -DGGML_CUDA=ON -DGGML_HIP=OFF -DGGML_VULKAN=OFF \
  -DCMAKE_CUDA_ARCHITECTURES='"$CUDA_ARCH"' > /dev/null
cmake --build /b -j "$(nproc)" --target llama-server --target llama-bench 2>&1 | tail -1
cp -a /b/bin/. /out/
# Bundle the CUDA 12 runtime libs the backend links against; the host only has 13.x.
for lib in libcudart libcublas libcublasLt libnccl; do
    src="$(ldconfig -p | grep -oE "/[^ ]*${lib}\.so\.[0-9]+" | head -1)"
    [ -n "$src" ] && cp -a "$src"* /out/ 2>/dev/null || true
done
' || die "Container build failed."

[[ -x "$WORK/out/llama-server" ]] || die "Build finished but llama-server missing."

OUT_CUDA="$LINUX_ROOT/runtime-cuda128"
rm -rf "$OUT_CUDA"; mkdir -p "$OUT_CUDA"
cp -a "$WORK/out/." "$OUT_CUDA/"
ok "CUDA 12.8 runtime: $OUT_CUDA"

# Merge with the local HIP backend for dual-vendor, if present. Same commit and
# GGML_BACKEND_DL on both sides is what makes the two ABI-compatible.
if [[ -x "$LINUX_ROOT/runtime-rocm-cuda/llama-server" ]]; then
    OUT_DUAL="$LINUX_ROOT/runtime-rocm-cuda128"
    rm -rf "$OUT_DUAL"; mkdir -p "$OUT_DUAL"
    cp -a "$LINUX_ROOT/runtime-rocm-cuda/." "$OUT_DUAL/"
    cp -f "$OUT_CUDA/libggml-cuda.so" "$OUT_DUAL/"
    for f in "$OUT_CUDA"/libcudart.so.* "$OUT_CUDA"/libcublas.so.* \
             "$OUT_CUDA"/libcublasLt.so.* "$OUT_CUDA"/libnccl.so.*; do
        [[ -e "$f" ]] && cp -a "$f" "$OUT_DUAL/"
    done
    ok "Dual runtime (CUDA 12.8 + local HIP): $OUT_DUAL"
    info "Launch with: ./start-llama-server.sh --mode rocm-cuda --runtime $OUT_DUAL -- ..."
else
    warn "runtime-rocm-cuda not found - built CUDA-only. Run setup-llama.sh --backend rocm-cuda first for a dual runtime."
fi

echo
info "Verify devices:"
LD_LIBRARY_PATH="${OUT_DUAL:-$OUT_CUDA}" "${OUT_DUAL:-$OUT_CUDA}/llama-server" --list-devices 2>/dev/null | sed -n '/Available devices/,$p'
