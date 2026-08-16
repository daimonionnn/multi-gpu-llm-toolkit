<#
.SYNOPSIS
    Test llama-server at 96 GB UMA with workarounds for KV cache spill.

.DESCRIPTION
    Runs a series of tests with increasing model/context sizes to find
    the boundary where KV cache spills from VRAM to shared memory at 96 GB UMA.

    Workarounds applied:
      - Quantized KV cache (--cache-type-k q8_0 --cache-type-v q8_0) to halve KV footprint
      - Flash attention enabled
      - GGML_CUDA_NO_PINNED=1 to avoid pinned memory failures
      - --cache-ram 0 to disable prompt cache (reduces VRAM pressure)
      - --fit off to prevent the memory estimator from spilling layers to CPU

.PARAMETER ModelPath
    Path to the GGUF model file to test.

.PARAMETER Context
    Context size to test with. Default: 4096.

.PARAMETER Port
    Server port. Default: 18081.

.PARAMETER KVQuant
    KV cache quantization type. Default: q8_0. Set to "f16" to disable quantization.

.PARAMETER NoKVQuant
    Disable KV cache quantization (use f16). For comparison testing.

.EXAMPLE
    # Test with a small model first
    .\test-96gb-uma.ps1 -ModelPath "C:\models\Qwen3.5-27B.Q4_K_M.gguf" -Context 4096

    # Test with quantized KV cache disabled (baseline comparison)
    .\test-96gb-uma.ps1 -ModelPath "C:\models\Qwen3.5-27B.Q4_K_M.gguf" -Context 4096 -NoKVQuant

    # Test with larger context
    .\test-96gb-uma.ps1 -ModelPath "C:\models\Qwen3.5-27B.Q4_K_M.gguf" -Context 32768
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [int]$Context = 4096,
    [int]$Port = 18081,
    [string]$KVQuant = "q8_0",
    [switch]$NoKVQuant,
    [string]$RuntimeDir = (Join-Path $PSScriptRoot "..\runtime-rocm-cuda")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ModelPath)) {
    throw "Model file not found: $ModelPath"
}
$serverExe = Join-Path $RuntimeDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    throw "llama-server.exe not found: $serverExe"
}

# Environment
$env:PATH = "$RuntimeDir;" + $env:PATH
$env:GGML_CUDA_NO_PINNED = "1"
Remove-Item Env:GGML_CUDA_ENABLE_UNIFIED_MEMORY -ErrorAction SilentlyContinue

$modelName = [System.IO.Path]::GetFileName($ModelPath)
$kvType = if ($NoKVQuant) { "f16" } else { $KVQuant }

Write-Host "=== 96 GB UMA Test ===" -ForegroundColor Cyan
Write-Host "  Model:     $modelName" -ForegroundColor Cyan
Write-Host "  Context:   $Context" -ForegroundColor Cyan
Write-Host "  KV cache:  $kvType" -ForegroundColor Cyan
Write-Host "  Port:      $Port" -ForegroundColor Cyan
Write-Host ""
Write-Host "Workarounds applied:" -ForegroundColor Yellow
Write-Host "  - GGML_CUDA_NO_PINNED=1" -ForegroundColor Yellow
Write-Host "  - --flash-attn on" -ForegroundColor Yellow
Write-Host "  - --cache-ram 0 (no prompt cache)" -ForegroundColor Yellow
Write-Host "  - --fit off (force GPU offload)" -ForegroundColor Yellow
if (-not $NoKVQuant) {
    Write-Host "  - --cache-type-k $kvType --cache-type-v $kvType (quantized KV)" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Watch for:" -ForegroundColor Magenta
Write-Host "  - 'ROCm_Host KV buffer' in output = KV cache spilled to shared memory (BAD)" -ForegroundColor Magenta
Write-Host "  - 'ROCm0 KV buffer' in output = KV cache in VRAM (GOOD)" -ForegroundColor Magenta
Write-Host "  - 'failed to allocate' = allocation failure" -ForegroundColor Magenta
Write-Host ""

$args = @(
    "-m", $ModelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "-c", "$Context",
    "--device", "ROCm0",
    "-ngl", "99",
    "--fit", "off",
    "--flash-attn", "on",
    "--cache-ram", "0",
    "--no-mmap"
)

if (-not $NoKVQuant) {
    $args += @("--cache-type-k", $kvType, "--cache-type-v", $kvType)
}

Write-Host "Command:" -ForegroundColor DarkGray
Write-Host "  $serverExe $($args -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& $serverExe @args
exit $LASTEXITCODE
