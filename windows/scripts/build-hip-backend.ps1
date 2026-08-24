# Build a HIP-only llama.cpp runtime for the AMD devices in this machine.
#
# Why this exists: upstream's Windows ROCm release zip is ABI-incompatible with
# the installed HIP SDK and enumerates no devices at all (doc/systems.md), and
# setup-llama.ps1 cannot help because every backend it knows about pulls in
# CUDA, which needs a toolkit this machine does not have. This builds just the
# AMD side, locally, against the installed SDK.
#
# Targets default to every AMD GPU present, read from hipInfo, so a machine with
# an APU and a discrete card gets both architectures in one library. A device
# whose gfx target is missing from the build still *enumerates* - it fails later
# when a kernel is launched - so getting this list right matters.
#
# Usage:
#   .\build-hip-backend.ps1                      # detect targets, build
#   .\build-hip-backend.ps1 -Targets gfx1151,gfx1201
#   .\build-hip-backend.ps1 -OutputDir ..\runtime-rocm-local

param(
    [string[]]$Targets,
    # Peer-to-peer copies between GPUs. Defaults to OFF (i.e. NO_PEER_COPY=ON)
    # because llama.cpp uses them whenever it splits a model across devices, and
    # on GPUs that are not peers - which `hipInfo` reports as
    # "non-peers: device#0 device#1" - the copies do not fail. They silently
    # transfer garbage, the server answers with token soup, and the timings look
    # normal. Verified on this machine: identical commit, identical ROCm
    # libraries, only this flag differs between a build that answers "Paris" and
    # one that answers "' 111t?/ 111t'". Turn it on only if your GPUs really are
    # peers and you want the bandwidth.
    [switch]$AllowPeerCopy,
    [string]$RepoDir,
    [string]$OutputDir,
    [string]$BuildDir = "build-hip",
    [int]$Jobs = 14
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not $RepoDir)   { $RepoDir   = Join-Path $projectRoot "llama.cpp" }
if (-not $OutputDir) { $OutputDir = Join-Path $projectRoot "runtime-rocm-local" }

$rocm = $env:HIP_PATH
if (-not $rocm) { $rocm = [Environment]::GetEnvironmentVariable("HIP_PATH", "Machine") }
if (-not $rocm -or -not (Test-Path $rocm)) { throw "HIP SDK not found. Set HIP_PATH." }
$rocm = $rocm.TrimEnd('\')

# Detect the gfx targets of every AMD device unless told otherwise.
if (-not $Targets) {
    $hipInfo = Join-Path $rocm "bin\hipInfo.exe"
    if (-not (Test-Path $hipInfo)) { throw "hipInfo.exe not found; pass -Targets explicitly." }
    $Targets = & $hipInfo 2>$null |
        Select-String -Pattern 'gcnArchName:\s+(gfx\w+)' |
        ForEach-Object { $_.Matches[0].Groups[1].Value } |
        Sort-Object -Unique
    if (-not $Targets) { throw "Could not detect any gfx target; pass -Targets explicitly." }
}
$targetList = $Targets -join ';'
Write-Host "HIP SDK:  $rocm"       -ForegroundColor Cyan
Write-Host "Targets:  $targetList" -ForegroundColor Cyan
Write-Host ("Peer copy: {0}" -f $(if ($AllowPeerCopy) { "enabled (risky on non-peer GPUs)" } else { "disabled" })) -ForegroundColor Cyan
Write-Host "Output:   $OutputDir"  -ForegroundColor Cyan

# ROCm's clang cannot link on its own - it needs the MSVC libraries, so import a
# developer environment first or it dies on msvcrtd.lib / oldnames.lib.
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) { throw "vswhere not found. Install Visual Studio Build Tools." }
$vsPath = (& $vsWhere -latest -property installationPath 2>$null)
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
cmd /c "`"$vcvars`" -vcvars_ver=14.44 >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}
$env:PATH = "$rocm\bin;$env:PATH"
$env:HIP_PATH = "$rocm\"

if (-not (Test-Path $RepoDir)) {
    git clone https://github.com/ggml-org/llama.cpp $RepoDir
}

# Two PowerShell traps in one line, both silent:
#   -DX=$(if (...) {...})  throws "the term 'if' is not recognized"
#   -DX=$var               passes the *literal* text $var, because a variable
#                          after '=' in an unquoted native-command argument is
#                          not expanded. CMake then treats the non-empty string
#                          as true, so the build happened to be correct for the
#                          wrong reason. Compute the value first, and quote it.
$noPeerCopy = if ($AllowPeerCopy) { "OFF" } else { "ON" }

Push-Location $RepoDir
try {
    $buildDir = $BuildDir
    cmake -S . -B $buildDir -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_C_COMPILER="$rocm/bin/clang.exe" `
        -DCMAKE_CXX_COMPILER="$rocm/bin/clang++.exe" `
        -DGGML_HIP=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF `
        -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF `
        -DAMDGPU_TARGETS="$targetList" -DGPU_TARGETS="$targetList" `
        -DGGML_CUDA_NO_PEER_COPY="$noPeerCopy" `
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }

    cmake --build $buildDir --target llama-server --target llama-bench -j $Jobs
    if ($LASTEXITCODE -ne 0) { throw "build failed" }

    $binDir = Join-Path $PWD "$buildDir\bin"
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Copy-Item "$binDir\*" -Destination $OutputDir -Recurse -Force
}
finally { Pop-Location }

Write-Host "Built: $OutputDir\llama-server.exe" -ForegroundColor Green
Write-Host "Note: run it with $rocm\bin on PATH (start-llama-server.ps1 does this for rocm modes)." -ForegroundColor Yellow
