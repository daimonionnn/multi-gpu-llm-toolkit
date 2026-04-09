param(
    [string]$RepoDir = (Join-Path $PSScriptRoot "..\llama.cpp"),
    [ValidateSet("rocm-cuda","vulkan","vulkan-cuda")]
    [string]$Backend = "rocm-cuda"
)

$ErrorActionPreference = "Stop"

# Choose runtime directory based on backend
switch ($Backend) {
    "vulkan"      { $RuntimeDir = Join-Path $PSScriptRoot "..\runtime-vulkan" }
    "vulkan-cuda" { $RuntimeDir = Join-Path $PSScriptRoot "..\runtime-vulkan-cuda" }
    default       { $RuntimeDir = Join-Path $PSScriptRoot "..\runtime-rocm-cuda" }        
}

# Prefer runtime binary because start scripts use runtime-rocm-cuda/llama-server.exe.     
$serverExe = Join-Path $RuntimeDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    $serverExe = Join-Path $RepoDir "build\bin\Release\llama-server.exe"
    if (-not (Test-Path $serverExe)) {
        $serverExe = Join-Path $RepoDir "build\bin\llama-server.exe"
    }
}

if (-not (Test-Path $serverExe)) {
    throw "llama-server.exe not found in runtime or build output. Run scripts/setup-llama.ps1 first."
}

$env:PATH = "$RuntimeDir;" + $env:PATH
Write-Host "Using: $serverExe  (backend: $Backend)" -ForegroundColor DarkCyan
& $serverExe --list-devices
