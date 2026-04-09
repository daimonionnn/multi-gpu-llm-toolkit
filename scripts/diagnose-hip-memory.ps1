<#
.SYNOPSIS
    Diagnose HIP memory state on AMD Strix Halo iGPU.
    Run this after changing BIOS UMA allocation (e.g. 64 GB vs 96 GB)
    to verify HIP reports correct free/total memory and that allocations succeed.

.DESCRIPTION
    1. Reports Windows physical memory via GlobalMemoryStatusEx
    2. Runs a Python HIP diagnostic that:
       - Reads hipMemGetInfo (free/total VRAM as seen by HIP)
       - Reads hipDeviceProperties (isLargeBar, totalGlobalMem)
       - Tests hipMalloc at increasing sizes to find the ceiling
       - Tests hipMallocManaged (expected broken on Strix Halo)
    3. Reports page file configuration

.NOTES
    Requires: amdhip64_7.dll in System32 (stock or patched), Python 3.x
#>

$ErrorActionPreference = "Stop"

Write-Host "=== HIP Memory Diagnostics ===" -ForegroundColor Cyan
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray
Write-Host ""

# --- Windows memory ---
Write-Host "--- Windows Physical Memory ---" -ForegroundColor Yellow
$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$freeGB  = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
Write-Host "  Total physical RAM : $totalGB GiB"
Write-Host "  Free physical RAM  : $freeGB GiB"
Write-Host ""

# --- Page file ---
Write-Host "--- Page File ---" -ForegroundColor Yellow
$pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
if ($pf) {
    foreach ($p in $pf) {
        $allocMB = $p.AllocatedBaseSize
        $usedMB  = $p.CurrentUsage
        Write-Host "  $($p.Name): $allocMB MiB allocated, $usedMB MiB used"
    }
} else {
    Write-Host "  No page file detected"
}
Write-Host ""

# --- HIP diagnostics via Python ---
Write-Host "--- HIP Device Memory ---" -ForegroundColor Yellow

$pyScript = @'
import ctypes, struct, sys

try:
    hip = ctypes.CDLL(r'C:\Windows\SYSTEM32\amdhip64_7.dll')
except OSError as e:
    print(f'  ERROR: Cannot load amdhip64_7.dll: {e}')
    sys.exit(1)

hip.hipSetDevice(0)

# Device count
count = ctypes.c_int(0)
hip.hipGetDeviceCount(ctypes.byref(count))
print(f'  HIP devices: {count.value}')

# hipMemGetInfo
free = ctypes.c_size_t(0)
total = ctypes.c_size_t(0)
err = hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
if err == 0:
    free_gb = free.value / (1024**3)
    total_gb = total.value / (1024**3)
    print(f'  hipMemGetInfo: free={free_gb:.2f} GiB, total={total_gb:.2f} GiB')
else:
    print(f'  hipMemGetInfo: FAILED (err={err})')

# Device properties
props = ctypes.create_string_buffer(4096)
err = hip.hipGetDeviceProperties(props, 0)
if err == 0:
    name = props.raw[:256].split(b'\x00')[0].decode()
    print(f'  Device name: {name}')

    # Find totalGlobalMem (size_t, scan likely offsets)
    for off in range(256, 512, 8):
        val = struct.unpack_from('<Q', props.raw, off)[0]
        if val > 30 * (1024**3) and val < 200 * (1024**3):
            print(f'  totalGlobalMem (offset {off}): {val / (1024**3):.2f} GiB')

    # isLargeBar scan
    for off in range(800, 1600, 4):
        val = struct.unpack_from('<i', props.raw, off)[0]
        if val in (0, 1):
            # Check neighboring fields for context
            prev = struct.unpack_from('<i', props.raw, off - 4)[0]
            next_val = struct.unpack_from('<i', props.raw, off + 4)[0]
            # isLargeBar is near isMultiGpuBoard (0 or 1) and other booleans
            if off == 1268:  # known offset in HIP SDK 7.1
                print(f'  isLargeBar (offset {off}): {val}')
else:
    print(f'  hipGetDeviceProperties: FAILED (err={err})')

print()
print('--- hipMalloc ceiling test ---')
last_ok = 0
first_fail = None
for size_gb in [20, 40, 50, 55, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96]:
    size = size_gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if err == 0:
        status = 'OK'
        last_ok = size_gb
        hip.hipFree(ptr)
    else:
        status = f'FAIL(err={err})'
        if first_fail is None:
            first_fail = size_gb
    print(f'  hipMalloc({size_gb:3d} GiB): {status}')

print()
if last_ok > 0:
    print(f'  => Max successful hipMalloc: {last_ok} GiB')
if first_fail:
    print(f'  => First failure at: {first_fail} GiB')

print()
print('--- hipMallocManaged test ---')
for size_gb in [1, 4, 16]:
    size = size_gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p(0)
    err = hip.hipMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(1))
    if err == 0:
        print(f'  hipMallocManaged({size_gb} GiB): OK')
        hip.hipFree(ptr)
    else:
        print(f'  hipMallocManaged({size_gb} GiB): FAIL(err={err})')
        break

print()
print('--- Sequential allocation test (KV cache simulation) ---')
# Allocate a large block (model) then a smaller block (KV cache) without freeing
model_gb = 40
kv_gb = 8
model_ptr = ctypes.c_void_p(0)
err1 = hip.hipMalloc(ctypes.byref(model_ptr), ctypes.c_size_t(model_gb * 1024**3))
if err1 == 0:
    print(f'  Model alloc ({model_gb} GiB): OK')
    kv_ptr = ctypes.c_void_p(0)
    err2 = hip.hipMalloc(ctypes.byref(kv_ptr), ctypes.c_size_t(kv_gb * 1024**3))
    if err2 == 0:
        print(f'  KV cache alloc ({kv_gb} GiB): OK  <- both in VRAM')
        hip.hipFree(kv_ptr)
    else:
        print(f'  KV cache alloc ({kv_gb} GiB): FAIL(err={err2})  <- KV may spill to shared memory')
    hip.hipFree(model_ptr)
else:
    print(f'  Model alloc ({model_gb} GiB): FAIL(err={err1})')

# Report post-test free memory
free2 = ctypes.c_size_t(0)
total2 = ctypes.c_size_t(0)
hip.hipMemGetInfo(ctypes.byref(free2), ctypes.byref(total2))
print(f'  Post-test hipMemGetInfo: free={free2.value / (1024**3):.2f} GiB')
'@

python -c $pyScript

Write-Host ""
Write-Host "=== Diagnostics complete ===" -ForegroundColor Cyan
