# Stop whatever this project is serving and hand the GPUs back.
#
# The recurring failure this exists for: a server is still resident, so the next
# launch either OOMs, or reports "no CUDA device detected" because the previous
# 146 GB model is still tearing down, or dies on a port something else owns. All
# three look like different bugs and are the same situation — something is still
# holding a resource.
#
# It stops **only this project's processes**, identified by the path they run
# from, not by their name. LM Studio ships an `llama-server.exe` of its own and
# so does anything else built on llama.cpp; those are reported and left running.
# Nothing here kills a process to free a port either — that is not this script's
# business.
#
# Usage:
#   .\stop-llama.ps1            stop this project's servers, wait for the VRAM
#   .\stop-llama.ps1 -Status    report only, change nothing
#   .\stop-llama.ps1 -All       also ask LM Studio to unload (opt-in)
#   .\stop-llama.ps1 -Help      this text (-h works too)

# PositionalBinding=$false is deliberate and load-bearing. Without it an
# unrecognised argument is either swallowed or bound to the first positional
# parameter, so `-h` ran the full stop and `--help` failed with "cannot convert
# '--help' to System.Int32" — asking a destructive script for help must never
# be the thing that fires it. Now anything unrecognised aborts before the first
# process is touched.
[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Status,
    [switch]$All,
    [Alias('h', '?')][switch]$Help,
    [int]$TimeoutSec = 300,
    [int[]]$Ports = @(8080, 8081, 8090, 8099)
)

if ($Help) {
    Get-Content $PSCommandPath | Select-Object -First 19 | ForEach-Object { $_ -replace '^#\s?', '' }
    exit 0
}

$ErrorActionPreference = "Continue"   # a cleanup script must finish its report

$projectRoot = Split-Path $PSScriptRoot -Parent      # ...\windows
$repoRoot    = Split-Path $projectRoot -Parent

function Write-Info { param($m) Write-Host "  $m" }
function Write-Head { param($m) Write-Host $m -ForegroundColor Cyan }

# ── Find our processes, and only ours ──────────────────────────────────
# Anything running out of this checkout counts, so runtimes added later are
# picked up without editing the script. Everything else is somebody's model
# host and gets left alone.
function Get-OurProcesses {
    $names = @('llama-server.exe', 'llama-bench.exe', 'llama-cli.exe', 'ggml-rpc-server.exe')
    $filter = ($names | ForEach-Object { "Name='$_'" }) -join ' OR '
    $all = @(Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue)

    $ours    = @($all | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($repoRoot, [StringComparison]::OrdinalIgnoreCase) })
    $foreign = @($all | Where-Object { $_ -notin $ours })
    return @{ Ours = $ours; Foreign = $foreign }
}

# Wrapper shells launched as `powershell -File ...\scripts\start-*.ps1`. They
# exit on their own once the server does, but a stuck one keeps a console and,
# more annoyingly, restarts nothing while looking like it might.
function Get-OurWrappers {
    @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine -match [regex]::Escape($PSScriptRoot) -and $_.CommandLine -notmatch 'stop-llama' })
}

# ── Reporting ──────────────────────────────────────────────────────────
function Show-Gpu {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $q = (nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>$null)
        foreach ($line in @($q)) {
            $f = $line -split ',\s*'
            if ($f.Count -ge 3) { Write-Info ("NVIDIA  {0,6} / {1,6} MiB   {2}" -f $f[1], $f[2], $f[0]) }
        }
        foreach ($app in @(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>$null)) {
            if ($app) { Write-Info ("        $app") }
        }
    }
    # Windows exposes no adapter *name* on these counters, only the LUID, so
    # they are printed as-is. The AMD iGPU is whichever one is not the NVIDIA
    # figure above; on a UMA machine its memory is system RAM either way.
    $mem = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples |
           Where-Object { $_.CookedValue -gt 500MB }
    foreach ($s in $mem) {
        $kind = if ($s.Path -match 'dedicated') { 'dedicated' } else { 'shared   ' }
        Write-Info ("adapter {0} {1} {2,6:N0} MiB" -f $s.InstanceName, $kind, ($s.CookedValue / 1MB))
    }
}

function Show-Ports {
    foreach ($p in $Ports) {
        $conn = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $conn) { continue }
        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.ProcessName } else { 'unknown' }
        Write-Info ("{0,-5} held by {1} (pid {2})" -f $p, $name, $conn.OwningProcess)
    }
}

$found = Get-OurProcesses

Write-Head "Before:"
Show-Gpu
Show-Ports
foreach ($p in $found.Ours)    { Write-Info ("ours:  {0} (pid {1})  {2}" -f $p.Name, $p.ProcessId, $p.ExecutablePath) }
foreach ($p in $found.Foreign) { Write-Host ("  other: {0} (pid {1}) - left alone  {2}" -f $p.Name, $p.ProcessId, $p.ExecutablePath) -ForegroundColor DarkGray }

if ($Status) { exit 0 }

# ── Stop them: politely, then not ──────────────────────────────────────
$targets = @($found.Ours) + @(Get-OurWrappers)
if ($targets.Count -eq 0) {
    Write-Host "Nothing of ours is running." -ForegroundColor Yellow
} else {
    Write-Head "Stopping $($targets.Count) process(es) ..."
    foreach ($p in $targets) {
        Write-Info ("taskkill {0} (pid {1})" -f $p.Name, $p.ProcessId)
        & taskkill.exe /PID $p.ProcessId 2>&1 | Out-Null
    }

    # Give them 20 s to exit on their own before forcing it. A 146 GB model
    # takes a few seconds to tear down even when it is cooperating.
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        $left = @(Get-OurProcesses).Ours
        if ($left.Count -eq 0) { break }
        Start-Sleep -Milliseconds 500
    }

    $left = @((Get-OurProcesses).Ours) + @(Get-OurWrappers)
    if ($left.Count -gt 0) {
        Write-Warning "Still alive after 20s, forcing"
        foreach ($p in $left) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}

# ── Optionally ask the other model hosts to unload ─────────────────────
# Opt-in only. The default must never touch LM Studio: on this rig it is the
# thing that downloaded the models in the first place.
if ($All) {
    if (Get-Command lms -ErrorAction SilentlyContinue) {
        Write-Head "Asking LM Studio to unload ..."
        & lms unload --all 2>&1 | Out-Null
    } else {
        Write-Info "lms CLI not found, nothing to ask"
    }
}

# ── Wait for the VRAM to actually come back ────────────────────────────
# The point of the whole script. Launching during teardown fails with
# "no CUDA device detected", which reads like a driver problem rather than the
# timing problem it is.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null) | Select-Object -First 1)
        if ($used -lt 4000) { break }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    if ($used -ge 4000) {
        Write-Warning "NVIDIA still holds $used MiB after ${TimeoutSec}s - see the process list below"
    }
}

Write-Host ""
Write-Head "After:"
Show-Gpu
Show-Ports

$remaining = @((Get-OurProcesses).Ours)
if ($remaining.Count -gt 0) {
    Write-Error "$($remaining.Count) of our process(es) still running"
    exit 1
}
Write-Host "GPUs released." -ForegroundColor Green
