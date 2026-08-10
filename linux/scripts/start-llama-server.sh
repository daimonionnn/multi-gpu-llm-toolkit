#!/usr/bin/env bash
# Main llama-server launcher. Model profile scripts call this.
# Linux port of start-llama-server.ps1.
#
# Modes:
#   rocm          — AMD single GPU (ROCm backend)
#   cuda          — NVIDIA single GPU (CUDA backend)
#   vulkan        — Single GPU via Vulkan (first detected)
#   rocm-cuda     — AMD ROCm + NVIDIA CUDA dual GPU
#   vulkan-vulkan — Both GPUs via Vulkan
#   vulkan-cuda   — AMD Vulkan + NVIDIA CUDA dual GPU
#
# Usage:
#   ./start-llama-server.sh --mode rocm-cuda [--port 8081] [--runtime DIR] \
#                           [--no-pinned] -- -m model.gguf -c 8192 -ngl 99
#
# Everything after `--` is passed through to llama-server unchanged.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE="rocm-cuda"
# 8081, not llama.cpp's usual 8080: on the dual-linux box a Caddy container
# from an unrelated docker stack publishes 8080->80 permanently, so a server
# started there dies with "couldn't bind HTTP server socket". The Windows
# scripts still default to 8080 - the clash is this machine's, not the
# project's, and there is nothing to gain by moving a port that works.
PORT=8081
RUNTIME_DIR=""
NO_PINNED=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="${2:?}"; shift 2 ;;
        --port)       PORT="${2:?}"; shift 2 ;;
        --runtime)    RUNTIME_DIR="${2:?}"; shift 2 ;;
        --no-pinned)  NO_PINNED=1; shift ;;
        -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
        --)           shift; EXTRA_ARGS=("$@"); break ;;
        *) die "Unknown argument: $1  (pass llama-server flags after --)" ;;
    esac
done

case "$MODE" in
    rocm|cuda|vulkan|rocm-cuda|vulkan-vulkan|vulkan-cuda) ;;
    *) die "Invalid mode '$MODE'." ;;
esac

# ── Resolve runtime directory ──────────────────────────────────────────
[[ -n "$RUNTIME_DIR" ]] || RUNTIME_DIR="$(runtime_dir_for "$MODE")"
SERVER_BIN="$RUNTIME_DIR/llama-server"
[[ -x "$SERVER_BIN" ]] || die "llama-server not found in $RUNTIME_DIR. Run setup-llama.sh first."

# ── Environment ────────────────────────────────────────────────────────
export_runtime_env "$RUNTIME_DIR"

# hipMallocManaged is broken on Strix Halo (gfx1151). Harmless to unset always.
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset VK_LOADER_DRIVERS_DISABLE

# NOTE — deliberate difference from Windows.
# start-llama-server.ps1 always sets GGML_CUDA_NO_PINNED=1 for ROCm/CUDA modes.
# That was a WDDM workaround; on Linux pinned host memory works normally and
# disabling it costs transfer bandwidth, so it is opt-in here via --no-pinned.
if (( NO_PINNED )); then
    export GGML_CUDA_NO_PINNED=1
    warn "GGML_CUDA_NO_PINNED=1 — pinned host memory disabled."
fi

# ── Device detection ───────────────────────────────────────────────────
device_text="$("$SERVER_BIN" --list-devices 2>&1 || true)"

cuda_dev="$(grep -oiE '\bCUDA[0-9]+\b'   <<< "$device_text" | head -1 || true)"
rocm_dev="$(grep -oiE '\bROCm[0-9]+\b'   <<< "$device_text" | head -1 || true)"
mapfile -t vulkan_devs < <(grep -oiE '\bVulkan[0-9]+\b' <<< "$device_text" || true)

# Selecting Vulkan devices by index is not safe on Linux. Unlike the Windows
# rig, a desktop Linux box typically also exposes an Intel/AMD iGPU and the
# llvmpipe software rasteriser through the same Vulkan loader — on dual-linux
# that is four Vulkan devices, only two of which are the cards we want. So
# match on the device description instead and keep the index order (AMD first,
# then NVIDIA) consistent with how rocm-cuda builds its device string.
vulkan_by_vendor() {
    grep -iE '^[[:space:]]*Vulkan[0-9]+:' <<< "$device_text" \
        | grep -iE "$1" | grep -oiE '\bVulkan[0-9]+\b' | head -1 || true
}
VK_AMD="$(vulkan_by_vendor 'amd|radeon')"
VK_NV="$(vulkan_by_vendor 'nvidia|geforce|rtx|quadro')"

# Anything that is neither, and is not a discrete card we asked for.
mapfile -t vk_other < <(
    grep -iE '^[[:space:]]*Vulkan[0-9]+:' <<< "$device_text" \
        | grep -iE 'llvmpipe|intel|swiftshader|software' \
        | grep -oiE '\bVulkan[0-9]+\b' || true
)
if [[ "$MODE" == vulkan* ]] && (( ${#vk_other[@]} > 0 )); then
    warn "Ignoring non-target Vulkan device(s): ${vk_other[*]} (iGPU or software rasteriser)"
fi

need() { [[ -n "$1" ]] || die "$MODE mode: $2"$'\n'"$device_text"; }

# Fall back to positional selection only if the description match failed.
vk_fallback() {
    local idx="$1" label="$2"
    (( ${#vulkan_devs[@]} > idx )) || die "$MODE mode: no Vulkan device for $label."$'\n'"$device_text"
    warn "Could not identify the $label Vulkan device by name; falling back to ${vulkan_devs[idx]}."
    echo "${vulkan_devs[idx]}"
}

case "$MODE" in
    rocm)
        need "$rocm_dev" "no ROCm device detected."
        DEVICE_ARG="$rocm_dev" ;;
    cuda)
        need "$cuda_dev" "no CUDA device detected."
        DEVICE_ARG="$cuda_dev" ;;
    vulkan)
        # Single-GPU Vulkan: prefer the AMD card, then NVIDIA, then whatever is first.
        DEVICE_ARG="${VK_AMD:-${VK_NV:-}}"
        [[ -n "$DEVICE_ARG" ]] || DEVICE_ARG="$(vk_fallback 0 'single-GPU')" ;;
    rocm-cuda)
        need "$rocm_dev" "need both ROCm and CUDA devices."
        need "$cuda_dev" "need both ROCm and CUDA devices."
        DEVICE_ARG="$rocm_dev,$cuda_dev" ;;
    vulkan-vulkan)
        amd="${VK_AMD:-$(vk_fallback 0 'AMD')}"
        nv="${VK_NV:-$(vk_fallback 1 'NVIDIA')}"
        [[ "$amd" != "$nv" ]] || die \
            "vulkan-vulkan mode: resolved both GPUs to the same device ($amd)."$'\n'"$device_text"
        DEVICE_ARG="$amd,$nv" ;;
    vulkan-cuda)
        amd="${VK_AMD:-$(vk_fallback 0 'AMD')}"
        need "$cuda_dev" "no CUDA device detected."
        DEVICE_ARG="$amd,$cuda_dev" ;;
esac

# ── Validate model path ────────────────────────────────────────────────
MODEL_PATH=""
for (( i = 0; i < ${#EXTRA_ARGS[@]}; i++ )); do
    if [[ "${EXTRA_ARGS[i]}" == "-m" || "${EXTRA_ARGS[i]}" == "--model" ]]; then
        MODEL_PATH="${EXTRA_ARGS[i+1]:-}"; break
    fi
done
if [[ -n "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    die "Model file not found: $MODEL_PATH"
fi

# ── Web UI ─────────────────────────────────────────────────────────────
# Since b10331 upstream ships the server web UI as a separate release asset
# (llama-<build>-ui.tar.gz) instead of embedding it; without it the server
# answers API calls but returns 415 on /. If the UI has been extracted to
# linux/webui/, serve it - unless the caller passed their own --path.
WEBUI_DIR="$LINUX_ROOT/webui"
if [[ -f "$WEBUI_DIR/index.html" ]]; then
    has_path=0
    for a in "${EXTRA_ARGS[@]}"; do [[ "$a" == "--path" ]] && { has_path=1; break; }; done
    (( has_path )) || EXTRA_ARGS+=(--path "$WEBUI_DIR")
fi

# ── Summary ────────────────────────────────────────────────────────────
info "llama-server  [$MODE]"
[[ -n "$MODEL_PATH" ]] && info "  Model:    $MODEL_PATH"
info "  Device:   $DEVICE_ARG"
info "  Runtime:  $RUNTIME_DIR"
info "  Port:     $PORT"
echo

exec "$SERVER_BIN" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --device "$DEVICE_ARG" \
    --webui \
    "${EXTRA_ARGS[@]}"
