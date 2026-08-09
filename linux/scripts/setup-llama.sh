#!/usr/bin/env bash
# Build llama.cpp for a given backend into runtime-<backend>/.
# Linux port of setup-llama.ps1.
#
# Usage:
#   ./setup-llama.sh [--backend rocm-cuda|vulkan|vulkan-cuda] [--repo DIR] [--out DIR] [--clean]

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BACKEND="rocm-cuda"
REPO_DIR="$LINUX_ROOT/llama.cpp"
OUTPUT_DIR=""
CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="${2:?}"; shift 2 ;;
        --repo)    REPO_DIR="${2:?}"; shift 2 ;;
        --out)     OUTPUT_DIR="${2:?}"; shift 2 ;;
        --clean)   CLEAN=1; shift ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

case "$BACKEND" in
    rocm|rocm-cuda|vulkan|vulkan-cuda) ;;
    *) die "Invalid backend '$BACKEND'. Use rocm, rocm-cuda, vulkan or vulkan-cuda." ;;
esac

[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$LINUX_ROOT/runtime-$BACKEND"

require_cmd git
require_cmd cmake
require_cmd ninja

uses_cuda=0; uses_hip=0; uses_vulkan=0
case "$BACKEND" in
    rocm)        uses_hip=1 ;;
    rocm-cuda)   uses_cuda=1; uses_hip=1 ;;
    vulkan)      uses_vulkan=1 ;;
    vulkan-cuda) uses_cuda=1; uses_vulkan=1 ;;
esac

cmake_flags=()

# ── CUDA ───────────────────────────────────────────────────────────────
if (( uses_cuda )); then
    arch="$(nvidia_arch)" || die "CUDA backend requested but nvidia-smi found no GPU."
    info "NVIDIA GPU compute capability: ${arch:0:${#arch}-1}.${arch: -1} (sm_$arch)"

    nvcc_bin="$(find_cuda_nvcc "$arch")" || die \
"No CUDA toolkit found that supports sm_$arch.
The nvcc first in PATH ($(command -v nvcc 2>/dev/null || echo none)) is too old for this GPU.
Install a newer CUDA Toolkit, or point CUDA_HOME at one that supports sm_$arch."

    cuda_root="$(dirname -- "$(dirname -- "$nvcc_bin")")"
    info "Using CUDA toolkit: $cuda_root ($("$nvcc_bin" --version | grep -oP 'release \K[0-9.]+'))"

    # Put the chosen toolkit ahead of any distro nvcc that would otherwise win.
    export CUDA_HOME="$cuda_root"
    export PATH="$cuda_root/bin:$PATH"
    cmake_flags+=("-DCMAKE_CUDA_COMPILER=$nvcc_bin" "-DCMAKE_CUDA_ARCHITECTURES=$arch")
fi

# ── ROCm / HIP ─────────────────────────────────────────────────────────
if (( uses_hip )); then
    rocm_info="$(detect_rocm)" || die \
"ROCm/HIP development files not found.
Install the HIP compiler and CMake config (Ubuntu: 'sudo apt install hipcc libhipblas-dev librocblas-dev'),
or build with --backend vulkan / vulkan-cuda instead."

    IFS='|' read -r HIPCXX ROCM_ROOT HIP_CMAKE_PREFIX <<< "$rocm_info"
    gfx="$(amd_gfx_target)" || die "Could not determine the AMD gfx target. Is rocminfo installed and the GPU visible?"

    info "ROCm root:    $ROCM_ROOT"
    info "HIP compiler: $HIPCXX"
    info "GPU target:   $gfx"

    export HIPCXX
    export CMAKE_PREFIX_PATH="$HIP_CMAKE_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

    # ROCM_PATH must only be set for a self-contained ROCm prefix such as
    # /opt/rocm. With the distro layout the root is /usr, and clang would then
    # look for device bitcode in /usr/amdgcn/bitcode, which does not exist —
    # the compiler check fails with "cannot find ROCm device library" even
    # though clang finds the libraries perfectly well on its own (they live in
    # its resource dir, /usr/lib/llvm-N/lib/clang/N/amdgcn/bitcode).
    if [[ "$ROCM_ROOT" != "/usr" ]]; then
        export ROCM_PATH="$ROCM_ROOT" HIP_PATH="$ROCM_ROOT"
    else
        info "Distro ROCm layout — leaving ROCM_PATH unset so clang uses its own device libs"
    fi
    cmake_flags+=("-DAMDGPU_TARGETS=$gfx" "-DGPU_TARGETS=$gfx")
fi

# ── Vulkan ─────────────────────────────────────────────────────────────
if (( uses_vulkan )); then
    command -v glslc >/dev/null 2>&1 || die \
"glslc (shader compiler) not found — the Vulkan backend cannot be built without it.
Ubuntu: sudo apt install glslc"
fi

# ── Backend flags ──────────────────────────────────────────────────────
case "$BACKEND" in
    rocm)
        info "Build: ROCm (HIP) backend only — AMD GPU, no NVIDIA"
        cmake_flags+=("-DGGML_HIP=ON" "-DGGML_CUDA=OFF" "-DGGML_VULKAN=OFF") ;;
    rocm-cuda)
        info "Build: ROCm (HIP) + CUDA backends"
        cmake_flags+=("-DGGML_HIP=ON" "-DGGML_CUDA=ON" "-DGGML_VULKAN=OFF") ;;
    vulkan)
        info "Build: Vulkan backend (drives both AMD and NVIDIA GPUs)"
        cmake_flags+=("-DGGML_VULKAN=ON" "-DGGML_CUDA=OFF" "-DGGML_HIP=OFF") ;;
    vulkan-cuda)
        info "Build: Vulkan (AMD) + CUDA (NVIDIA) backends"
        cmake_flags+=("-DGGML_VULKAN=ON" "-DGGML_CUDA=ON" "-DGGML_HIP=OFF") ;;
esac

# ── Source ─────────────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR" ]]; then
    info "Cloning llama.cpp into $REPO_DIR ..."
    git clone https://github.com/ggml-org/llama.cpp "$REPO_DIR"
else
    info "Updating $REPO_DIR ..."
    git -C "$REPO_DIR" pull --ff-only
fi

BUILD_DIR="$REPO_DIR/build-$BACKEND"

# A stale cache remembers the old compiler paths, which is exactly what changes
# when the toolkit selection above picks a different nvcc.
if (( CLEAN )) || { (( uses_cuda )) && [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; }; then
    [[ -d "$BUILD_DIR" ]] && { warn "Removing stale build directory $BUILD_DIR"; rm -rf "$BUILD_DIR"; }
fi

# ── Configure and build ────────────────────────────────────────────────
cmake -S "$REPO_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    "${cmake_flags[@]}"

cmake --build "$BUILD_DIR" -j "$(jobs_count)" --target llama-server --target llama-bench

# ── Collect runtime ────────────────────────────────────────────────────
server_bin="$BUILD_DIR/bin/llama-server"
[[ -x "$server_bin" ]] || die "Build finished but llama-server was not found at $server_bin"

mkdir -p "$OUTPUT_DIR"
info "Copying build output to $OUTPUT_DIR ..."
cp -a "$BUILD_DIR/bin/." "$OUTPUT_DIR/"

# With GGML_BACKEND_DL the ggml backends are separate shared objects; depending
# on the version they land beside the binaries or under ggml/src.
find "$BUILD_DIR" -name 'libggml*.so*' -type f -exec cp -a {} "$OUTPUT_DIR/" \; 2>/dev/null || true

ok "Built ($BACKEND): $OUTPUT_DIR/llama-server"
echo
info "Backends present in the runtime directory:"
ls -1 "$OUTPUT_DIR" | grep -E '^libggml-.*\.so' | sed 's/^/  /' || echo "  (none — check the build log)"
echo
info "Next step:"
info "  ./start-llama-server.sh --mode $BACKEND -- -m /path/to/model.gguf -ngl 99"
