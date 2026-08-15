# DeepSeek V4 Flash 0731 MXFP4 (146 GB) - the lossless reference quant.
#
# The model is QAT with native MXFP4 experts, so this quant IS the original
# weights; every other quant is equal at best (Q8) or lossy (Q4 and below).
#
# Rig: halo-win - Strix Halo 395 + RTX PRO 6000 96 GB, external over OCuLink.
# Two facts about that rig decide this profile, and both invert the tuning that
# `linux/scripts/start-deepseek-mxfp4-nvidia-amd-cpu.sh` uses on `dual-linux`:
#
#   1. The NVIDIA card is four lanes wide (PCIe 4.0 x4, ~8 GB/s, against
#      PCIe 5.0 x16 on the Linux rig). Anything that streams expert weights
#      across it during prefill is starved: CUDA-only with --n-cpu-moe 18
#      measures 299 pp here against 1175 pp on `dual-linux` with identical
#      flags. That is the lane count, not the cabling - the same layout over a
#      Thunderbolt 5 tunnel gave 261 pp, so going native bought 15%, not 4x.
#   2. The AMD iGPU's memory *is* system RAM, so putting the overflow experts
#      on it costs no bus traffic at all. Nothing needs --n-cpu-moe, and the
#      146 GB model ends up fully GPU-resident: ~89 GB on CUDA0, ~61 GB on the
#      iGPU.
#
# Measured 2026-08-15, -c 131072, 1 GB BIOS framebuffer, pp/tg at 16k context
# (full matrix and mechanisms in doc/benchmarks.md):
#
#   -Layout rocm-cuda    497.5 pp / 36.13 tg   default; needs a local ggml-hip.dll
#   -Layout vulkan-cuda  352.1 pp / 34.35 tg   prebuilt only, no HIP build needed
#   -Layout cuda-only    299.1 pp / 22.93 tg   no iGPU; needs ~56 GB of free RAM
#
# Leave the BIOS iGPU framebuffer at its minimum. A 64 GB carve-out is not fully
# CPU-visible (isLargeBar: 0), which costs Vulkan 45% of its prefill and HIP
# ~6%, and it leaves the OS too little RAM for the cuda-only layout.
#
# Only the expert FFN tensors may go to the iGPU. A classic layer split
# (`-ts 22,21`, no `-ot`) puts attention on the iGPU too, and llama.cpp then
# disables all four fused DeepSeek V4 ops it cannot run there -
# "Lightning Indexer / HC pre / HC comb / HC post not supported, set to
# disabled" - before dying during warmup. With `-ot` those warnings do not
# appear at all.
#
# Usage:
#   .\start-deepseek-mxfp4-nvidia-amd.ps1                        # rocm-cuda dual, 128k
#   .\start-deepseek-mxfp4-nvidia-amd.ps1 -Layout vulkan-cuda    # no HIP build needed
#   .\start-deepseek-mxfp4-nvidia-amd.ps1 -Layout cuda-only      # no iGPU at all
#
# Serves an OpenAI-compatible API and llama.cpp's own web UI on the same port.
#   .\start-deepseek-mxfp4-nvidia-amd.ps1 -Port 8091 -ExtraArgs @('--api-key','x')

param(
    [ValidateSet("rocm-cuda", "vulkan-cuda", "cuda-only")]
    [string]$Layout = "rocm-cuda",

    [int]$Port = 8090,          # 8080 is taken by AgentService on this machine
    [string]$RuntimeDir,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$scriptDir   = $PSScriptRoot
$projectRoot = Split-Path $scriptDir -Parent

# GGUF root; override with LLM_MODELS_DIR. Default is LM Studio's download dir.
$modelsDir = if ($env:LLM_MODELS_DIR) { $env:LLM_MODELS_DIR } else { Join-Path $env:USERPROFILE ".lmstudio\models" }
$model = Join-Path $modelsDir "lmstudio-community\DeepSeek-V4-Flash-0731-GGUF\DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf"

if (-not (Test-Path $model)) {
    Write-Error @"
Model not found: $model
Download it first, e.g. in LM Studio search 'lmstudio-community DeepSeek-V4-Flash-0731 MXFP4'.
Or set LLM_MODELS_DIR if your GGUFs live outside $modelsDir.
"@
    exit 1
}

# Runtimes here are upstream prebuilt b10441 binaries with backend DLLs mixed by
# hand, not local builds - there is no CUDA Toolkit on this machine, and none is
# needed. See windows/README.md, "Runtimes without a compiler". CUDA 12.4 was
# measured against 13.3 and is identical within noise on this model (DeepSeek's
# MLA path does not hit the Blackwell FA collapse of doc/cuda-fa-blackwell.md),
# so 13.3 wins on having native sm_120 cubins and no PTX JIT on first request.
#
# 18 layers of experts (blk 25-42) is the VRAM ceiling at -c 131072: CUDA0 sits
# at 95.5 of 97.9 GB and 17 layers no longer loads. The split is the strongest
# lever measured - each layer moved back to CUDA0 is worth ~4.5% pp and ~2% tg.
$expertRegex = 'blk\.(2[5-9]|3[0-9]|4[0-2])\.ffn_.*_exps.*'

switch ($Layout) {
    "rocm-cuda" {
        # Needs ggml-hip.dll built locally for gfx1151 - the upstream ROCm zip
        # is ABI incompatible with HIP SDK 7.1 and enumerates no devices.
        $mode        = "rocm-cuda"
        $defaultRt   = "runtime-rocm-cuda133"
        $placement   = @("-ts", "0,1", "-ot", "$expertRegex=ROCm0")
    }
    "vulkan-cuda" {
        $mode        = "vulkan-cuda"
        $defaultRt   = "runtime-vulkan-cuda133"
        $placement   = @("-ts", "0,1", "-ot", "$expertRegex=Vulkan0")
    }
    "cuda-only" {
        # Fallback with no iGPU: 18 expert layers stream from system RAM across
        # the TB5 tunnel. Slower on both axes, but it is the layout to use if
        # the AMD path proves unstable - on `dual-linux` the equivalent AMD path
        # faulted under DeepSeek (doc/benchmarks.md), which has not been
        # reproduced here yet. Will not fit if the BIOS framebuffer is large.
        $mode        = "cuda"
        $defaultRt   = "runtime-cuda133"
        $placement   = @("--n-cpu-moe", "18")
    }
}

if (-not $RuntimeDir) { $RuntimeDir = Join-Path $projectRoot $defaultRt }

# -b 4096 -ub 2048 rather than llama.cpp's 2048/512, but for a weaker reason
# than on `dual-linux`, where the bigger micro-batch was worth +55-60% of
# prefill because it amortised expert weight transfers over PCIe. Here nothing
# streams, so there is nothing to amortise: -b 8192 -ub 4096 measured no better,
# and its extra compute buffers cost an expert layer on CUDA0, which is a net
# loss. Keep 4096/2048 and spend spare VRAM on expert layers instead.
#
# -lm none (the old --no-mmap) because the 146 GB file does not fit the page
# cache; mmap would page CPU-side tensors back off NVMe.
$serverArgs = @(
    "-m", $model,
    "-c", "131072",
    "-ngl", "99"
) + $placement + @(
    "-fa", "on",
    "-b", "4096", "-ub", "2048",
    "-lm", "none",
    "--jinja"
) + $ExtraArgs

# Pinned host memory has to be decided per layout, and both settings break the
# other layout:
#
# - The dual layouts need it ON. The launcher's default for CUDA/HIP modes is
#   GGML_CUDA_NO_PINNED=1, and with a 64 GB BIOS carve-out that leaves the OS
#   63.6 GB, unpinned host staging collides with the 57.4 GiB single allocation
#   the iGPU needs: the load either fails ("alloc_tensor_range: failed to
#   allocate ROCm0 buffer of size 61605937152") or hangs at exactly that point.
#
# - cuda-only needs it OFF. Its RAM-hosted experts are one ~58.4 GiB host
#   buffer, and pinning that much fails outright ("failed to allocate CUDA_Host
#   buffer of size 62664998912") even with 112 GB free - it is the size of the
#   single pinned allocation, not the amount of free memory. Unpinned costs
#   host-to-device transfer speed, which this layout leans on, so the number to
#   expect is the one measured with the flag off.
$pinned = @{}
if ($Layout -ne "cuda-only") { $pinned["AllowPinned"] = $true }

& (Join-Path $scriptDir "start-llama-server.ps1") `
    -Mode $mode `
    -Port $Port `
    -RuntimeDir $RuntimeDir `
    @pinned `
    -ExtraArgs $serverArgs
