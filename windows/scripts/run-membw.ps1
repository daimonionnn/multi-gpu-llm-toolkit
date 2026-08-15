# Build and run membw.c — the streaming-read bandwidth benchmark that prices CPU
# expert offload. See doc/performance-model.md for what the number is used for.
#
# Prints the same line as linux/scripts/membw.c so the rigs compare directly:
#   read bandwidth: 89.9 GB/s (16 threads, 64 GB in 0.71 s)
#
# Needs <threads> GB of free RAM (16 GB by default). Nothing else may be using
# the memory bus — stop llama-server first, or the number is meaningless.

param(
    [int]$Threads = 16,
    [int]$Runs = 3
)

$ErrorActionPreference = "Stop"

$src = Join-Path $PSScriptRoot "membw.c"
$exe = Join-Path $env:TEMP "membw.exe"

function Initialize-VcVars64 {
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsWhere)) { return $false }
    $vsPath = (& $vsWhere -latest -property installationPath 2>$null)
    if (-not $vsPath) { return $false }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { return $false }
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
    return [bool](Get-Command cl.exe -ErrorAction SilentlyContinue)
}

# Prefer MSVC; fall back to any clang on PATH (the HIP SDK ships one).
$built = $false
if ((Get-Command cl.exe -ErrorAction SilentlyContinue) -or (Initialize-VcVars64)) {
    Push-Location $env:TEMP
    try {
        & cl.exe /nologo /O2 /MT $src /Fe:$exe | Out-Null
        $built = $LASTEXITCODE -eq 0
    } finally { Pop-Location }
}
if (-not $built -and (Get-Command clang.exe -ErrorAction SilentlyContinue)) {
    & clang.exe -O2 $src -o $exe
    $built = $LASTEXITCODE -eq 0
}
if (-not $built) {
    throw "No usable C compiler. Install VS 2022 Build Tools (Desktop development with C++), or put clang on PATH."
}

if (Get-Process llama-server -ErrorAction SilentlyContinue) {
    Write-Warning "llama-server is running; it will skew the result. Stop it for a clean number."
}

$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
Write-Host "Free RAM: $free GB, benchmark needs $Threads GB" -ForegroundColor Cyan

$results = 1..$Runs | ForEach-Object {
    $line = & $exe $Threads
    Write-Host "  run $_`: $line"
    [double]([regex]::Match($line, 'bandwidth: ([\d.]+)').Groups[1].Value)
}

$best = ($results | Measure-Object -Maximum).Maximum
$avg  = ($results | Measure-Object -Average).Average
Write-Host ("best {0:N1} GB/s, mean {1:N1} GB/s over {2} runs" -f $best, $avg, $Runs) -ForegroundColor Green
