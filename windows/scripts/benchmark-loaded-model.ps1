param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [int[]]$Contexts = @(1024, 4096, 16384),
    [int]$PredictTokens = 256,
    [int]$TimeoutSec = 7200,
    [string]$OutCsv = "",
    [string]$Mode = "",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http

function Invoke-JsonPost {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][object]$Payload,
        [int]$Timeout = 120
    )

    $body = $Payload | ConvertTo-Json -Depth 10 -Compress
    return Invoke-RestMethod -Method Post -Uri $Url -ContentType "application/json" -Body $body -TimeoutSec $Timeout
}

function Get-TokenCount {
    param(
        [Parameter(Mandatory = $true)][string]$Base,
        [Parameter(Mandatory = $true)][string]$Text,
        [int]$Timeout = 120
    )

    try {
        $resp = Invoke-JsonPost -Url "$Base/tokenize" -Payload @{ content = $Text } -Timeout $Timeout

        if ($null -ne $resp.tokens) {
            return [int]$resp.tokens.Count
        }

        if ($resp -is [System.Array]) {
            return [int]$resp.Count
        }

        return [int][Math]::Ceiling($Text.Length / 4.0)
    }
    catch {
        return [int][Math]::Ceiling($Text.Length / 4.0)
    }
}

function New-BenchmarkPrompt {
    param(
        [Parameter(Mandatory = $true)][string]$Base,
        [Parameter(Mandatory = $true)][int]$TargetTokens,
        [int]$Timeout = 120
    )

    $chunk = "The quick brown fox jumps over the lazy dog near the river bank. "
    $chunkTokens = Get-TokenCount -Base $Base -Text $chunk -Timeout $Timeout
    if ($chunkTokens -lt 1) { $chunkTokens = 12 }

    $repeatCount = [Math]::Max(1, [int][Math]::Ceiling($TargetTokens / $chunkTokens))
    $prompt = ($chunk * $repeatCount).Trim()
    $actualTokens = Get-TokenCount -Base $Base -Text $prompt -Timeout $Timeout

    $guard = 0
    while ($actualTokens -lt $TargetTokens -and $guard -lt 4) {
        $missing = $TargetTokens - $actualTokens
        $more = [Math]::Max(1, [int][Math]::Ceiling($missing / $chunkTokens))
        $prompt = ($prompt + " " + ($chunk * $more)).Trim()
        $actualTokens = Get-TokenCount -Base $Base -Text $prompt -Timeout $Timeout
        $guard++
    }

    return [PSCustomObject]@{
        Prompt = $prompt
        Tokens = $actualTokens
    }
}

function Get-LoadedModel {
    param([string]$Base)

    try {
        $models = Invoke-RestMethod -Method Get -Uri "$Base/v1/models" -TimeoutSec 15
        if ($models.data.Count -gt 0 -and $models.data[0].id) {
            return [string]$models.data[0].id
        }
    }
    catch {
    }

    return "unknown"
}

try {
    $health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 10
    if ($health.status -ne "ok") {
        throw "Server health check returned unexpected response: $($health | ConvertTo-Json -Compress)"
    }
}
catch {
    throw "Cannot reach llama-server at $BaseUrl. Start your model first, then rerun. Error: $($_.Exception.Message)"
}

$modelName = Get-LoadedModel -Base $BaseUrl
Write-Host "Benchmarking loaded model: $modelName" -ForegroundColor Cyan
Write-Host "Server: $BaseUrl" -ForegroundColor Cyan
Write-Host "Contexts: $($Contexts -join ', ')" -ForegroundColor Cyan
Write-Host "Predict tokens per run: $PredictTokens" -ForegroundColor Cyan
Write-Host ""

$cmdLineArgsStr = $ExtraArgs -join ' '
try {
    $llamaProc = Get-CimInstance Win32_Process -Filter "Name LIKE '%llama-server%'" | Select-Object -First 1
    if ($llamaProc -and $llamaProc.CommandLine) {
        $cmdLine = $llamaProc.CommandLine
        if ([string]::IsNullOrWhiteSpace($Mode) -and $cmdLine -match "runtime-([^\\]+)\\llama-server") {
            $Mode = $matches[1]
        }
        if ([string]::IsNullOrWhiteSpace($cmdLineArgsStr)) {
            $cmdLineArgsStr = $cmdLine -replace '^"[^"]+"\s*', '' -replace '^[^ ]+\s*', ''
            $ExtraArgs = [System.Text.RegularExpressions.Regex]::Matches($cmdLineArgsStr, '("[^"]*"|\S+)').Value
            $cmdLineArgsStr = $ExtraArgs -join ' '
        }
    }
} catch { }

$maxParallel = 1
for ($i = 0; $i -lt $ExtraArgs.Length; $i++) {
    if ($ExtraArgs[$i] -eq "--parallel" -and $i -lt ($ExtraArgs.Length - 1)) {
        [int]$parsed = 0
        if ([int]::TryParse($ExtraArgs[$i+1].Trim('"'), [ref]$parsed)) {
            $maxParallel = $parsed
        }
    }
}

$results = New-Object System.Collections.Generic.List[object]

foreach ($ctx in $Contexts) {
    $targetPromptTokens = [Math]::Max(256, $ctx - $PredictTokens - 32)
    Write-Host "Preparing prompt for context $ctx (target prompt tokens: ~$targetPromptTokens)..." -ForegroundColor Yellow

    $promptObj = New-BenchmarkPrompt -Base $BaseUrl -TargetTokens $targetPromptTokens -Timeout $TimeoutSec
    $prompt = $promptObj.Prompt
    $promptTokens = [int]$promptObj.Tokens

    $payload = @{
        prompt = $prompt
        n_predict = $PredictTokens
        temperature = 0.1
        top_p = 0.95
        top_k = 40
        cache_prompt = $false
        stream = $false
    }
    $payloadJson = $payload | ConvertTo-Json -Depth 10 -Compress

    for ($p = 1; $p -le $maxParallel; $p++) {
        Write-Host "Running inference for context $ctx with prompt tokens: $promptTokens (Parallel Requests: $p)" -ForegroundColor Yellow

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $tasks = New-Object System.Collections.Generic.List[System.Threading.Tasks.Task]
        $client = New-Object System.Net.Http.HttpClient
        $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)

        for ($k = 1; $k -le $p; $k++) {
            $content = New-Object System.Net.Http.StringContent($payloadJson, [System.Text.Encoding]::UTF8, "application/json")
            $tasks.Add($client.PostAsync("$BaseUrl/completion", $content))
        }

        try { [System.Threading.Tasks.Task]::WaitAll($tasks.ToArray()) } catch { }
        $sw.Stop()
        $TotalWallMs = $sw.Elapsed.TotalMilliseconds

        $totalGenTok = 0
        $totalGenTps = 0
        $totalPrefillTok = 0
        $totalPrefillTps = 0
        $successes = 0
        $errMsgs = @()

        foreach ($t in $tasks) {
            if ($t.Status -eq 'RanToCompletion') {
                try {
                    $jsonStr = $t.Result.Content.ReadAsStringAsync().Result
                    $resp = $jsonStr | ConvertFrom-Json
                    
                    if ($resp.timings) {
                        $successes++
                        $totalGenTok += $resp.timings.predicted_n
                        $totalGenTps += $resp.timings.predicted_per_second
                        $totalPrefillTok += $resp.timings.prompt_n
                        $totalPrefillTps += $resp.timings.prompt_per_second
                    } else {
                        $errMsgs += "invalid response"
                    }
                } catch {
                    $errMsgs += $_.Exception.Message
                }
            } else {
                $err = if ($t.Exception) { $t.Exception.InnerException.Message } else { "request failed or timed out" }
                $errMsgs += $err
            }
        }
        $client.Dispose()
        $avgGenTps = if ($successes -gt 0) { $totalGenTps / $successes } else { 0 }
        $avgPrefillTps = if ($successes -gt 0) { $totalPrefillTps / $successes } else { 0 }
        $avgPromptTok = if ($successes -gt 0) { $totalPrefillTok / $successes } else { $promptTokens }
        $avgPredictTok = if ($successes -gt 0) { $totalGenTok / $successes } else { 0 }
        $totTokPerSec = if ($TotalWallMs -gt 0) { ($totalGenTok) / ($TotalWallMs/1000.0) } else { 0 }
        $Status = if ($errMsgs.Count -eq 0) { "ok" } else { "failed: $($errMsgs -join ', ')" }

        $results.Add([PSCustomObject]@{
            Context = $ctx
            ParallelReqs = $p
            PromptTokens = [math]::Round($avgPromptTok)
            PredictTokens = [math]::Round($avgPredictTok)        
            PrefillTokPerSec = [math]::Round($avgPrefillTps, 2)
            GenTokPerSec = [math]::Round($avgGenTps, 2)
            TotalTokPerSec = [math]::Round($totTokPerSec, 2)
            TotalWallMs = [math]::Round($TotalWallMs, 2)
            Status = $Status
        })
    }
    Write-Host ""
}

Write-Host "Benchmark results:" -ForegroundColor Green
$results | Format-Table Context, ParallelReqs, PromptTokens, PredictTokens, PrefillTokPerSec, GenTokPerSec, TotalTokPerSec, TotalWallMs, Status -AutoSize

if (-not [string]::IsNullOrWhiteSpace($OutCsv)) {
    $results | Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8
    Write-Host "CSV written to: $OutCsv" -ForegroundColor Green
}

$logPath = Join-Path (Split-Path -Parent $PSCommandPath) "benchmark.log"
$logContent = @"
==========================================
Date: $(Get-Date)
Model: $modelName
Mode: $Mode
ExtraArgs: $( $ExtraArgs -join ' ' )
Contexts: $( $Contexts -join ', ' )
Predict tokens: $PredictTokens

"@
$logContent += ($results | Format-Table Context, ParallelReqs, PromptTokens, PredictTokens, PrefillTokPerSec, GenTokPerSec, TotalTokPerSec, TotalWallMs, Status -AutoSize | Out-String)
$logContent += "`n"
Add-Content -Path $logPath -Value $logContent
Write-Host "Benchmark log written to: $logPath" -ForegroundColor Green
