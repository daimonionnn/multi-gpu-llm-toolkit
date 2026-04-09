param(
    [Parameter(Mandatory = $false)]
    [string]$Mode = "",
    [string[]]$ExtraArgs = @(),
    [int[]]$PromptTokens = @(128, 512, 1024),
    [int[]]$PredictTokens = @(128, 256),
    [int[]]$BatchSizes = @(1, 2, 4)
)

$ErrorActionPreference = "Stop"

# If no arguments provided, try to detect them from a running llama-server process
if ([string]::IsNullOrWhiteSpace($Mode) -and $ExtraArgs.Count -eq 0) {
    try {
        $llamaProc = Get-CimInstance Win32_Process -Filter "Name LIKE '%llama-server%'" | Select-Object -First 1
        if ($llamaProc -and $llamaProc.CommandLine) {
            Write-Host "Warning: llama-server is currently running!" -ForegroundColor Yellow
            Write-Host "Running llama-bench.exe with the same model concurrently may cause Out of Memory (OOM) errors." -ForegroundColor Yellow
            
            $cmdLine = $llamaProc.CommandLine
            if ($cmdLine -match "runtime-([^\\]+)\\llama-server") {
                $Mode = $matches[1]
            }
            $cmdLineArgsStr = $cmdLine -replace '^"[^"]+"\s*', '' -replace '^[^ ]+\s*', ''
            $allArgs = [System.Text.RegularExpressions.Regex]::Matches($cmdLineArgsStr, '("[^"]*"|\S+)').Value
            
            # Filter out server-specific args that llama-bench doesn't support
            $serverOnlyArgs = @("--host", "--port", "--webui", "--parallel")
            $filteredArgs = New-Object System.Collections.Generic.List[string]
            $skipNext = $false

            for ($i = 0; $i -lt $allArgs.Length; $i++) {
                if ($skipNext) { $skipNext = $false; continue }
                
                $arg = $allArgs[$i].Trim('"')
                if ($serverOnlyArgs -contains $arg) {
                    if ($arg -match "^--host|--port|--parallel$") { $skipNext = $true }
                    continue
                }
                $filteredArgs.Add($allArgs[$i])
            }
            $ExtraArgs = $filteredArgs.ToArray()
        } else {
            Write-Host "llama-server is not running and no arguments were provided." -ForegroundColor Red
            Write-Host "Either start your server first to auto-detect arguments, or pass -Mode and -ExtraArgs manually." -ForegroundColor Cyan
            exit 1
        }
    } catch {
        Write-Host "Failed to query running processes. Provide -Mode and -ExtraArgs manually." -ForegroundColor Red
        exit 1
    }
}

$basePath = Join-Path (Split-Path -Parent $PSCommandPath) ".."
$benchPath = Join-Path $basePath "runtime-$Mode\llama-bench.exe"

if (-not (Test-Path $benchPath)) {
    Write-Host "Cannot find llama-bench at $benchPath" -ForegroundColor Red
    Write-Host "It looks like llama-bench.exe was not built for the '$Mode' runtime." -ForegroundColor Yellow
    Write-Host "To fix this, you may need to update your build script (setup-llama.ps1) to add '--target llama-bench' alongside '--target llama-server' so that it gets compiled and copied into the runtime folder." -ForegroundColor Cyan
    exit 1
}

# Format parameters for llama-bench
$pStr = $PromptTokens -join ','
$nStr = $PredictTokens -join ','
$bStr = $BatchSizes -join ','

$benchArgs = @("-p", $pStr, "-n", $nStr, "-b", $bStr)
$benchArgs += $ExtraArgs

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Starting llama-bench hardware test    " -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor White
Write-Host "Base Args: $($ExtraArgs -join ' ')" -ForegroundColor White
Write-Host "Tests: Prompts ($pStr), Predict ($nStr), Batch Sizes ($bStr)" -ForegroundColor White
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

try {
    $proc = Start-Process -NoNewWindow -Wait -PassThru -FilePath $benchPath -ArgumentList $benchArgs
    if ($proc.ExitCode -ne 0) {
        Write-Host "llama-bench exited with code $($proc.ExitCode)" -ForegroundColor Red
    } else {
        Write-Host "llama-bench finished successfully." -ForegroundColor Green
    }
} catch {
    Write-Host "Failed to execute llama-bench.exe" -ForegroundColor Red
    Write-Host $_.Exception.Message
}