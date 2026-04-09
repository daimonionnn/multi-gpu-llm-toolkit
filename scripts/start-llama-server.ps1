# Main llama-server launcher.
# Model-specific scripts call this with their parameters.
#
# Modes:
#   rocm          — AMD single GPU (ROCm backend)
#   cuda          — NVIDIA single GPU (CUDA backend)
#   vulkan        — Single GPU via Vulkan (first detected)
#   rocm-cuda     — AMD ROCm + NVIDIA CUDA dual GPU
#   vulkan-vulkan — Both GPUs via Vulkan
#   vulkan-cuda   — AMD Vulkan + NVIDIA CUDA dual GPU

param(
    [ValidateSet("rocm", "cuda", "vulkan", "rocm-cuda", "vulkan-vulkan", "vulkan-cuda")]
    [string]$Mode = "rocm-cuda",

    [int]$Port         = 8080,
    [string]$RuntimeDir,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

# ── Resolve runtime directory ──────────────────────────────────────────
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not $RuntimeDir) {
    switch -Wildcard ($Mode) {
        "vulkan-cuda" {
            $candidate = Join-Path $projectRoot "runtime-vulkan-cuda"
            if (Test-Path (Join-Path $candidate "llama-server.exe")) {
                $RuntimeDir = $candidate
            } else {
                $RuntimeDir = Join-Path $projectRoot "runtime-vulkan"
            }
            break
        }
        "vulkan*" {
            $RuntimeDir = Join-Path $projectRoot "runtime-vulkan"
            break
        }
        default {
            $RuntimeDir = Join-Path $projectRoot "runtime-rocm-cuda"
        }
    }
}

$serverExe = Join-Path $RuntimeDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    throw "llama-server.exe not found in $RuntimeDir. Run setup-llama.ps1 first."
}

# ── Environment ────────────────────────────────────────────────────────
$env:PATH = "$RuntimeDir;" + $env:PATH

if ($Mode -in @("rocm", "cuda", "rocm-cuda")) {
    $env:GGML_CUDA_NO_PINNED = "1"
    Remove-Item Env:GGML_CUDA_ENABLE_UNIFIED_MEMORY -ErrorAction SilentlyContinue
}

# Clean up any leftover Vulkan driver filter
Remove-Item Env:VK_LOADER_DRIVERS_DISABLE -ErrorAction SilentlyContinue

# ── Device detection ──────────────────────────────────────────────────
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
$deviceText = & $serverExe --list-devices 2>&1 | Out-String
$ErrorActionPreference = $prevEAP
$cudaMatch     = [regex]::Match($deviceText, "(?im)\b(CUDA\d+)\b")
$rocmMatch     = [regex]::Match($deviceText, "(?im)\b(ROCm\d+)\b")
$vulkanMatches = [regex]::Matches($deviceText, "(?im)\b(Vulkan\d+)\b")

$isDual = $Mode -in @("rocm-cuda", "vulkan-vulkan", "vulkan-cuda")

switch ($Mode) {
    "rocm" {
        if (-not $rocmMatch.Success) { throw "rocm mode: no ROCm device detected.`n$deviceText" }
        $deviceArg = $rocmMatch.Groups[1].Value
    }
    "cuda" {
        if (-not $cudaMatch.Success) { throw "cuda mode: no CUDA device detected.`n$deviceText" }
        $deviceArg = $cudaMatch.Groups[1].Value
    }
    "vulkan" {
        if ($vulkanMatches.Count -eq 0) { throw "vulkan mode: no Vulkan device detected.`n$deviceText" }
        $deviceArg = $vulkanMatches[0].Groups[1].Value
    }
    "rocm-cuda" {
        if (-not $rocmMatch.Success -or -not $cudaMatch.Success) {
            throw "rocm-cuda mode: need both ROCm and CUDA devices.`n$deviceText"
        }
        $deviceArg = "$($rocmMatch.Groups[1].Value),$($cudaMatch.Groups[1].Value)"
    }
    "vulkan-vulkan" {
        if ($vulkanMatches.Count -lt 2) {
            throw "vulkan-vulkan mode: need at least 2 Vulkan devices, found $($vulkanMatches.Count).`n$deviceText"
        }
        $deviceArg = "$($vulkanMatches[0].Groups[1].Value),$($vulkanMatches[1].Groups[1].Value)"
    }
    "vulkan-cuda" {
        if ($vulkanMatches.Count -eq 0) { throw "vulkan-cuda mode: no Vulkan device detected.`n$deviceText" }
        if (-not $cudaMatch.Success)    { throw "vulkan-cuda mode: no CUDA device detected.`n$deviceText" }
        $deviceArg = "$($vulkanMatches[0].Groups[1].Value),$($cudaMatch.Groups[1].Value)"
    }
}

# ── Build argument list ───────────────────────────────────────────────
$serverArgs = @(
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--device", $deviceArg,
    "--webui"
)

if ($ExtraArgs.Count -gt 0) {
    $serverArgs += $ExtraArgs
}

# Extract model path from ExtraArgs for validation and display
$ModelPath = $null
for ($i = 0; $i -lt $ExtraArgs.Count; $i++) {
    if ($ExtraArgs[$i] -eq "-m" -and $i + 1 -lt $ExtraArgs.Count) {
        $ModelPath = $ExtraArgs[$i + 1]; break
    }
}
if ($ModelPath -and -not (Test-Path $ModelPath)) {
    throw "Model file not found: $ModelPath"
}

# ── Summary ───────────────────────────────────────────────────────────
Write-Host "llama-server  [$Mode]" -ForegroundColor Cyan
if ($ModelPath) {
    Write-Host "  Model:    $ModelPath" -ForegroundColor Cyan
}
Write-Host "  Device:   $deviceArg" -ForegroundColor Cyan
Write-Host "  Port:     $Port" -ForegroundColor Cyan
Write-Host ""

& $serverExe @serverArgs
exit $LASTEXITCODE
