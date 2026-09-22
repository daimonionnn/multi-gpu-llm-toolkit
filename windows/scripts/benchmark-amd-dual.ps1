# Benchmark a model across the AMD devices in this machine: the Strix Halo iGPU
# alone, the discrete card alone, and both together.
#
# The question it answers: what is a discrete AMD card worth next to an APU
# whose "VRAM" is system RAM? The iGPU has capacity (112 GB here) but shares the
# LPDDR5X bus; the discrete card has a tenth of the capacity and its own much
# faster VRAM, reached over whatever link the dock provides. Which side wins
# depends entirely on whether the model fits the small fast pool.
#
# Devices are matched by description, never by index: ROCm and Vulkan enumerate
# them in opposite orders on this machine (ROCm0 is the discrete card, Vulkan0
# is the iGPU), so an index-based script would silently benchmark the wrong
# thing.
#
# Usage:
#   .\benchmark-amd-dual.ps1 -Model D:\models\foo.gguf
#   .\benchmark-amd-dual.ps1 -Model ... -Configs igpu,dual -Contexts 4096,16384
#   .\benchmark-amd-dual.ps1 -Model ... -TensorSplit 1,3        # dual only
#
# Each config is a full load/measure/unload cycle, so a large model takes a few
# minutes per row. Results are printed as one table and written to the CSV,
# which is **overwritten** each run - a second run with -Configs dual loses the
# rows a first run with -Configs igpu wrote. Pass -OutCsv to keep them apart.

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [ValidateSet('igpu', 'discrete', 'dual')][string[]]$Configs = @('igpu', 'discrete', 'dual'),
    [int[]]$Contexts = @(4096, 16384),
    [ValidateSet('rocm', 'vulkan')][string]$Backend = 'rocm',
    [string]$Runtime,
    [string]$TensorSplit,
    # Correctness gate: every config must answer this before it is timed.
    [string]$ProbePrompt = "Question: What is the capital city of France? Answer with the city name only.`nAnswer:",
    [string]$ProbeExpect = 'Paris',
    [int]$PredictTokens = 128,
    # Token budget for the correctness probe. A reasoning model spends its first
    # tokens inside a <think> block and has not reached the answer by token 12,
    # which the gate then records as CORRUPT OUTPUT. Raise it for those models;
    # the think block is stripped before matching either way.
    [int]$ProbeTokens = 12,
    [int]$Port = 8090,
    [int]$LoadTimeoutSec = 900,
    [string]$OutCsv,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not $Runtime) {
    $Runtime = Join-Path $projectRoot $(if ($Backend -eq 'vulkan') { "runtime-vulkan" } else { "runtime-rocm-local" })
}
if (-not $OutCsv)  { $OutCsv  = Join-Path $projectRoot ("logs\bench-amd-dual-{0}.csv" -f [IO.Path]::GetFileNameWithoutExtension($Model)) }
$serverExe = Join-Path $Runtime "llama-server.exe"
if (-not (Test-Path $serverExe)) { throw "llama-server.exe not found in $Runtime - build it with build-hip-backend.ps1" }
if (-not (Test-Path $Model))     { throw "Model not found: $Model" }

$hip = $env:HIP_PATH; if (-not $hip) { $hip = [Environment]::GetEnvironmentVariable("HIP_PATH", "Machine") }
if ($hip) { $env:PATH = (Join-Path $hip.TrimEnd('\') 'bin') + ";$env:PATH" }
$env:PATH = "$Runtime;$env:PATH"

# ---- resolve devices by description -----------------------------------
$listing = & $serverExe --list-devices 2>&1 | Out-String
$devRe   = if ($Backend -eq 'vulkan') { 'Vulkan\d+' } else { 'ROCm\d+' }
$igpuDev = ([regex]::Match($listing, "(?im)^\s*($devRe):\s*.*(8060S|Radeon\(TM\) 8\d{3})")).Groups[1].Value
$discDev = ([regex]::Match($listing, "(?im)^\s*($devRe):\s*.*(R9700|AI PRO|W7\d{3}|RX \d{4})")).Groups[1].Value
if (-not $igpuDev) { throw "Could not find the integrated GPU in:`n$listing" }
Write-Host "iGPU device:     $igpuDev" -ForegroundColor Cyan
Write-Host "discrete device: $(if ($discDev) { $discDev } else { '(none found)' })" -ForegroundColor Cyan
Write-Host ("model:           {0} ({1:N1} GB)" -f $Model, ((Get-Item $Model).Length / 1GB)) -ForegroundColor Cyan

function Stop-Server {
    Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($projectRoot, [StringComparison]::OrdinalIgnoreCase) } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 4
}

$results = New-Object System.Collections.Generic.List[object]

foreach ($cfg in $Configs) {
    if ($cfg -eq 'discrete' -and -not $discDev) { Write-Warning "skipping 'discrete': no discrete AMD device"; continue }
    if ($cfg -eq 'dual'     -and -not $discDev) { Write-Warning "skipping 'dual': no discrete AMD device";     continue }

    $device = switch ($cfg) {
        'igpu'     { $igpuDev }
        'discrete' { $discDev }
        'dual'     { "$igpuDev,$discDev" }
    }
    $maxCtx = ($Contexts | Measure-Object -Maximum).Maximum

    Stop-Server
    $log = Join-Path $projectRoot ("logs\bench-{0}-{1}.log" -f $cfg, [IO.Path]::GetFileNameWithoutExtension($Model))
    $sargs = @('--host', '127.0.0.1', '--port', "$Port", '--device', $device,
               '-m', $Model, '-c', "$maxCtx", '-ngl', '99', '-fa', 'on', '--jinja')
    if ($cfg -eq 'dual' -and $TensorSplit) { $sargs += @('-ts', $TensorSplit) }
    $sargs += $ExtraArgs

    Write-Host "`n=== $cfg ($device) ===" -ForegroundColor Yellow
    $proc = Start-Process -FilePath $serverExe -ArgumentList $sargs -PassThru -WindowStyle Hidden `
                          -RedirectStandardOutput $log -RedirectStandardError "$log.err"

    $deadline = (Get-Date).AddSeconds($LoadTimeoutSec)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        try {
            if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).status -eq 'ok') { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 5
    }
    if (-not $ready) {
        $why = (Select-String -Path "$log.err" -Pattern 'error|failed to' -CaseSensitive:$false |
                Select-Object -Last 1).Line
        Write-Warning "$cfg did not come up: $why"
        $results.Add([PSCustomObject]@{ Config = "$Backend/$cfg"; Context = '-'; PrefillTokPerSec = 0; GenTokPerSec = 0; Status = "load failed: $why" })
        Stop-Server
        continue
    }

    # Correctness gate. Splitting a model across this machine's two AMD GPUs with
    # the ROCm backend loads, runs, reports plausible timings - and emits token
    # soup. A benchmark that only measures speed records those numbers as if they
    # meant something, so every config has to answer one factual question first.
    $probeBody = @{ prompt = $ProbePrompt; n_predict = $ProbeTokens; temperature = 0; cache_prompt = $false } | ConvertTo-Json -Compress
    $probeOut = ''
    try {
        $probeOut = (Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/completion" `
                        -ContentType 'application/json' -Body $probeBody -TimeoutSec 600).content
    } catch { $probeOut = "<request failed: $($_.Exception.Message)>" }

    # Strip the reasoning block before matching, including an unterminated one:
    # a model still thinking when the budget ran out has no </think> at all.
    $probeAnswer = $probeOut -replace '(?s)<think>.*?</think>', ''
    $probeAnswer = $probeAnswer -replace '(?s)<think>.*$', ''

    if ($probeAnswer -notmatch [regex]::Escape($ProbeExpect)) {
        $short = ($probeOut -replace '\s+', ' ').Trim()
        if ($short.Length -gt 60) { $short = $short.Substring(0, 60) + '...' }
        Write-Warning "$cfg produces WRONG OUTPUT - not timing it. Expected '$ProbeExpect', got: $short"
        $results.Add([PSCustomObject]@{ Config = "$Backend/$cfg"; Context = 0; PromptTokens = 0; PrefillTokPerSec = 0; GenTokPerSec = 0
                                        Status = "CORRUPT OUTPUT: $short" })
        Stop-Server
        continue
    }
    Write-Host "  correctness probe: ok" -ForegroundColor DarkGray

    # benchmark-loaded-model.ps1 prints a formatted table rather than emitting
    # objects, so take its results through the CSV it writes instead of trying
    # to capture the pipeline - captured format records have no usable fields.
    $tmpCsv = Join-Path $env:TEMP ("bench-{0}-{1}.csv" -f $cfg, [guid]::NewGuid().ToString('N').Substring(0,8))
    & (Join-Path $PSScriptRoot 'benchmark-loaded-model.ps1') `
        -BaseUrl "http://127.0.0.1:$Port" -Contexts $Contexts -PredictTokens $PredictTokens `
        -Mode $cfg -OutCsv $tmpCsv | Out-Null
    if (Test-Path $tmpCsv) {
        foreach ($row in (Import-Csv $tmpCsv)) {
            $results.Add([PSCustomObject]@{
                Config           = "$Backend/$cfg"
                Context          = [int]$row.Context
                PromptTokens     = [int]$row.PromptTokens
                PrefillTokPerSec = [double]$row.PrefillTokPerSec
                GenTokPerSec     = [double]$row.GenTokPerSec
                Status           = $row.Status
            })
        }
        Remove-Item $tmpCsv -ErrorAction SilentlyContinue
    } else {
        $results.Add([PSCustomObject]@{ Config = "$Backend/$cfg"; Context = 0; PrefillTokPerSec = 0; GenTokPerSec = 0; Status = 'benchmark produced no CSV' })
    }
    Stop-Server
}

Write-Host "`nResults: $Model" -ForegroundColor Green
$results | Format-Table Config, Context, PromptTokens, PrefillTokPerSec, GenTokPerSec, Status -AutoSize
$results | Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8
Write-Host "CSV: $OutCsv" -ForegroundColor Green
