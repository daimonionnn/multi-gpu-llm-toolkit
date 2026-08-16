param(
    [string]$RepoDir = (Join-Path $PSScriptRoot "..\llama.cpp"),

    [ValidateSet("rocm-cuda", "vulkan", "vulkan-cuda")]
    [string]$Backend = "rocm-cuda",

    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' is not in PATH."
    }
}

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][scriptblock]$Script)
    & $Script
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Initialize-VcVars64 {
    param([string]$VcVarsVer)
    # Find vcvars64.bat via vswhere
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsWhere)) {
        throw "vswhere not found. Install Visual Studio Build Tools."
    }
    $vsPath = (& $vsWhere -latest -property installationPath 2>$null)
    if (-not $vsPath) { throw "No Visual Studio installation found." }

    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at: $vcvars" }

    $vcvarsArgs = if ($VcVarsVer) { "`"$vcvars`" -vcvars_ver=$VcVarsVer" } else { "`"$vcvars`"" }
    Write-Host "Initializing x64 developer environment (toolset $VcVarsVer) ..." -ForegroundColor Yellow

    # Run vcvars64.bat and capture the resulting environment
    $envLines = cmd /c "$vcvarsArgs >nul 2>&1 && set"
    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

function Find-CudaCompatibleToolset {
    # CUDA 13.x cudafe++ crashes with MSVC v14.50+ (VS 18 / v145 toolset).
    # Returns the toolset version string (e.g. "14.44") to pass to vcvars64 -vcvars_ver, or $null if current is fine.
    try {
        $clOutput = (& cl.exe 2>&1 | Out-String)
    } catch {
        $clOutput = $_.Exception.Message
    }

    # Check if we need a workaround: cl.exe >= 19.50 or not x64
    $needsFix = $false
    if ($clOutput -match 'Version\s+(\d+)\.(\d+)') {
        $clMajor = [int]$Matches[1]
        $clMinor = [int]$Matches[2]
        if ($clMajor -gt 19 -or ($clMajor -eq 19 -and $clMinor -ge 50)) { $needsFix = $true }
    }
    if ($clOutput -match 'for x86') { $needsFix = $true }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { $needsFix = $true }

    if (-not $needsFix) { return $null }

    Write-Host "Current compiler is incompatible with CUDA (too new or x86) - searching for compatible MSVC toolset..." -ForegroundColor Yellow

    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsWhere)) { throw "vswhere not found." }

    $vsPaths = @(& $vsWhere -all -property installationPath 2>$null)
    foreach ($vsPath in $vsPaths) {
        $msvcRoot = Join-Path $vsPath "VC\Tools\MSVC"
        if (-not (Test-Path $msvcRoot)) { continue }

        $compatible = Get-ChildItem $msvcRoot -Directory |
            Where-Object { try { [version]$_.Name -lt [version]"14.50.0" } catch { $false } } |
            Sort-Object { [version]$_.Name } -Descending |
            Select-Object -First 1

        if ($compatible) {
            # Extract major.minor for -vcvars_ver (e.g. "14.44.35207" -> "14.44")
            $ver = [version]$compatible.Name
            $vcvarsVer = "$($ver.Major).$($ver.Minor)"
            Write-Host "Found compatible MSVC toolset $($compatible.Name)" -ForegroundColor Green
            return $vcvarsVer
        }
    }

    throw 'No MSVC toolset older than 14.50 found. Install the MSVC v143 (VS 2022 C++ x64/x86 build tools) component via the Visual Studio Installer.'
}

Require-Command git
Require-Command cmake

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue) -and -not (Get-Command clang++.exe -ErrorAction SilentlyContinue)) {
    throw "No C/C++ compiler found. Install Visual Studio 2022 Build Tools (Desktop development with C++) and run this from a Developer PowerShell."
}

if (-not (Test-Path $RepoDir)) {
    git clone https://github.com/ggml-org/llama.cpp $RepoDir
}

# Determine output directory name based on backend
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $projectRoot = Split-Path $PSScriptRoot -Parent
    $OutputDir = Join-Path $projectRoot "runtime-$Backend"
}

# Select CMake flags based on backend
$usesCuda = $Backend -in @("rocm-cuda", "vulkan-cuda")
$usesHip = $Backend -eq "rocm-cuda"

if ($usesCuda) {
    Require-Command nvcc
    $compatVer = Find-CudaCompatibleToolset
    if ($compatVer) {
        # Re-initialize the entire x64 dev environment with the compatible toolset
        Initialize-VcVars64 -VcVarsVer $compatVer
    }
}

# Ensure HIP SDK is discoverable for ROCm builds
if ($usesHip) {
    $hipPath = $env:HIP_PATH
    if (-not $hipPath) {
        $hipPath = [System.Environment]::GetEnvironmentVariable("HIP_PATH", "Machine")
        if (-not $hipPath) { $hipPath = [System.Environment]::GetEnvironmentVariable("HIP_PATH", "User") }
    }
    
    if ($hipPath -and (Test-Path $hipPath)) {
        $env:HIP_PATH = $hipPath
        # Tell CMake where to find HIP SDK packages (including hipblas, etc)
        $hipCmakeDir = Join-Path $hipPath "lib\cmake"
        if (Test-Path $hipCmakeDir) {
            $env:CMAKE_PREFIX_PATH = if ($env:CMAKE_PREFIX_PATH) { "$hipCmakeDir;$env:CMAKE_PREFIX_PATH" } else { $hipCmakeDir }
            Write-Host "Added HIP SDK to CMake search path: $hipCmakeDir" -ForegroundColor Cyan
        }
    } else {
        throw "HIP SDK not found. Install AMD HIP SDK and ensure HIP_PATH is set, or use -Backend vulkan / vulkan-cuda instead."
    }
}
switch ($Backend) {
    "rocm-cuda" {
        Write-Host "Build: ROCm (HIP) + CUDA backends" -ForegroundColor Cyan
        $cmakeFlags = @("-DGGML_HIP=ON", "-DGGML_CUDA=ON", "-DGGML_VULKAN=OFF")
    }
    "vulkan" {
        Write-Host "Build: Vulkan backend (handles both AMD + NVIDIA GPUs)" -ForegroundColor Cyan
        $cmakeFlags = @("-DGGML_VULKAN=ON", "-DGGML_CUDA=OFF", "-DGGML_HIP=OFF")
    }
    "vulkan-cuda" {
        Write-Host "Build: Vulkan (AMD) + CUDA (NVIDIA) backends" -ForegroundColor Cyan
        $cmakeFlags = @("-DGGML_VULKAN=ON", "-DGGML_CUDA=ON", "-DGGML_HIP=OFF")
    }
}

Push-Location $RepoDir
try {
    Invoke-Checked { git pull --ff-only }

    $buildDir = "build-$Backend"

    # Always clean stale CMake cache for CUDA builds (toolset environment may have changed)
    if ($usesCuda -and (Test-Path (Join-Path $buildDir "CMakeCache.txt"))) {
        Write-Host "Removing stale CMake cache in $buildDir ..." -ForegroundColor Yellow
        Remove-Item $buildDir -Recurse -Force
    }

    $allFlags = @("-S", ".", "-B", $buildDir, "-G", "Ninja", "-DGGML_BACKEND_DL=ON", "-DGGML_NATIVE=OFF") + $cmakeFlags
    Invoke-Checked { cmake @allFlags }
    Invoke-Checked { cmake --build $buildDir --config Release -j 12 --target llama-server --target llama-bench }

    # Find built executable
    $serverExe = Join-Path $PWD "$buildDir\bin\Release\llama-server.exe"
    if (-not (Test-Path $serverExe)) {
        $serverExe = Join-Path $PWD "$buildDir\bin\llama-server.exe"
    }

    if (-not (Test-Path $serverExe)) {
        throw "Build completed but llama-server.exe was not found in expected output paths."
    }

    # Copy build output to runtime directory
    $binDir = Split-Path $serverExe -Parent
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    Write-Host "Copying build output to $OutputDir ..." -ForegroundColor Cyan
    Copy-Item "$binDir\*" -Destination $OutputDir -Recurse -Force

    # Also copy backend DLLs from build root (ggml-*.dll)
    $buildRoot = Join-Path $PWD $buildDir
    Get-ChildItem "$buildRoot\ggml\src\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination $OutputDir -Force
    }
    # Some builds put DLLs next to the binary
    Get-ChildItem "$binDir\ggml-*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination $OutputDir -Force
    }

    Write-Host "Built server ($Backend): $OutputDir\llama-server.exe" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host ('  .\scripts\start-llama-server.ps1 -Mode {0} -ExtraArgs @(''-m'',''your-model.gguf'',''-ngl'',''99'')' -f $Backend) -ForegroundColor Yellow
}
finally {
    Pop-Location
}
