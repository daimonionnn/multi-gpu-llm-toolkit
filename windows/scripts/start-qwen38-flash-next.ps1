# Qwen3.8-Flash-Next Q4_K_M (111 GiB, arch qwen4exp) - dual AMD profile
#
# 512 experts / 10 active, 48 blocks, hybrid SSM + sparse attention, per-layer
# embeddings. Measured on halo-win, ROCm, b11065 local HIP build, -fa on:
#
#   layout            ctx      pp (4 runs)            tg (4 runs)
#   dual  -ts 2,1     4096     538.8 - 558.5          21.5 - 25.8
#   dual  -ts 2,1    16384     563.0 - 576.0 (+364)   20.1 - 21.9
#   iGPU alone        4096     389.5 / 401.1           5.86 / 5.74
#   iGPU alone       16384     371.5 / 378.8           5.63 / 5.46
#
# Prefill is the tight measurement: 3.7% spread at 4k, 2.3% at 16k once the
# single 364.0 outlier is set aside - it sits 36% below three runs that agree,
# and was the only one measured on the full-rate link, which is backwards.
# Prefill also *rises* from 4k to 16k in three runs of four, which nothing else
# in doc/benchmarks.md does. Generation is looser: quote 3.7x-4.4x over the
# iGPU as a range, not a point.
#
# The result contradicts the rule of thumb in doc/benchmarks.md, that a model
# fitting the iGPU should stay there - and the reason is specific to this
# model, see (3).
#
# Five things decide this layout, and none of them is the file size.
#
# 1. `per_layer_token_embd.weight` is 32.78 GiB of **F32** - Q4_K_M does not
#    quantize it - and llama.cpp keeps it in host memory, like Gemma 3n's PLE.
#    It shares one host buffer with `token_embd.weight`: 33.12 GiB together,
#    so 77.8 GiB of the 111 GiB file is what actually reaches a GPU.
#
# 2. HIP charges GPU allocations against the Windows **commit limit**, the iGPU
#    carve-out and the discrete card's own VRAM included. 77.8 GiB of buffers
#    plus the 33.12 GiB host buffer needs ~136 GiB of commit. With the stock 64 GiB
#    pagefile the limit is 127.6 GiB and the load dies at
#    `failed to allocate ROCm_Host buffer of size 35557749760` - at every
#    tensor split, because moving weights between devices does not change the
#    total. This rig runs a **96 GiB pagefile with InitialSize = MaximumSize**;
#    a lazily-grown pagefile does not expand fast enough to serve one 33 GiB
#    request. Physical RAM is never the constraint - it stayed at 48 GiB free
#    throughout the failures.
#
# 3. The iGPU *can* hold all 77.8 GiB alone, and doing so is four times slower
#    at generation. Past its BIOS carve-out the overflow lives in GTT, so those
#    weights and the PLE table then read from the same LPDDR5X bus, per token.
#    Moving a third of the weights into the R9700's own GDDR6 splits that
#    traffic across two memory systems. Capacity is not the reason to go dual
#    here; bus contention is.
#
# 4. Vulkan cannot run this model on this rig at all. The R9700 exposes a
#    256 MiB device-local + host-visible heap (small BAR, because OCuLink needs
#    Resizable BAR off to enumerate), and the Vulkan backend asks that heap for
#    a ~955 MiB buffer, at every split tried, in both device orders.
#
# 5. A flaky Thunderbolt dock shows up as
#      ROCm error: unspecified launch failure
#      ... in ggml_backend_cuda_buffer_set_tensor, current device: -1
#    around 30 s into the weight upload, and looks exactly like a llama.cpp bug.
#    It is not: re-seating the dock made the identical command work. If dual
#    dies there, power-cycle the dock before changing anything in software.
#
#    And it comes back. **Eight of twelve dual loads failed**, all at 22-27 s.
#    Renegotiating the link down to 2.5 GT/s made it worse, not better (0 of 3),
#    so this is not the dock failing to hold a high rate; reinstalling the same
#    driver version gave the best rate seen (2 of 3). The layout is unreliable,
#    the numbers it produces are not: -Layout igpu has never failed and
#    reproduces inside 3% if you need a load that just works.
#
# Needs a runtime that knows `qwen4exp`: nothing before upstream b11065 does,
# and upstream's ROCm zip enumerates no device against HIP SDK 7.1. Build one:
#   .\build-hip-backend.ps1 -Targets gfx1151,gfx1201 -BuildDir build-hip-b11065 `
#                           -OutputDir ..\runtime-rocm-b11065-local
#
# Usage:
#   .\start-qwen38-flash-next.ps1                  # both GPUs, 4k context
#   .\start-qwen38-flash-next.ps1 -Context 32768
#   .\start-qwen38-flash-next.ps1 -Layout igpu     # iGPU alone, dock not needed
#   .\start-qwen38-flash-next.ps1 -Vision          # also load the mmproj

param(
    [ValidateSet("dual", "igpu")]
    [string]$Layout = "dual",

    [int]$Port = 8090,          # 8080 is taken by AgentService on this machine
    [int]$Context = 4096,

    # 2,1 puts ~26 GiB on the R9700, inside its 31.6 GiB, and the rest on the
    # iGPU. Measured against 1.5,1; both load, 2,1 is the one benchmarked above.
    [string]$TensorSplit = "2,1",

    [string]$RuntimeDir,
    [string]$Devices,
    [switch]$Vision,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeDir) { $RuntimeDir = Join-Path $projectRoot "runtime-rocm-b11065-local" }

# ROCm enumerates the discrete card first on this machine - ROCm0 is the R9700,
# ROCm1 the 8060S iGPU. Vulkan orders them the other way round, which is why
# this is spelled out rather than left to an index.
if (-not $Devices) { $Devices = if ($Layout -eq "dual") { "ROCm1,ROCm0" } else { "ROCm1" } }

# GGUF root; override with $env:LLM_MODELS_DIR. Default is LM Studio's download dir.
$ModelsDir = if ($env:LLM_MODELS_DIR) { $env:LLM_MODELS_DIR } else { "$env:USERPROFILE\.lmstudio\models" }
$modelDir  = Join-Path $ModelsDir "lmstudio-community\Qwen3.8-Flash-Next-GGUF"
$model     = Join-Path $modelDir  "Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf"
$mmproj    = Join-Path $modelDir  "mmproj-Qwen3.8-Flash-Next-BF16.gguf"

# --fit off because the fitter mis-projects this model: left on, it moves a
# further ~33 GiB of weights into a host buffer on top of the embedding one, and the
# load dies before it reads a byte.
$modelArgs = @(
    "-m", $model,
    "-c", "$Context",
    "-ngl", "99",
    "--flash-attn", "on",
    "--fit", "off",
    "--jinja"
)

if ($Layout -eq "dual") {
    $modelArgs += @("--split-mode", "layer", "--tensor-split", $TensorSplit)
}

if ($Vision) {
    if (-not (Test-Path $mmproj)) { throw "mmproj not found: $mmproj" }
    $modelArgs += @("--mmproj", $mmproj)
}

$modelArgs += $ExtraArgs

& "$PSScriptRoot\start-llama-server.ps1" `
    -Mode rocm `
    -Port $Port `
    -RuntimeDir $RuntimeDir `
    -Devices $Devices `
    -ExtraArgs $modelArgs
