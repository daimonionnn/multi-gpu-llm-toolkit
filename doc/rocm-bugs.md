# ROCm Bugs Affecting Strix Halo Memory

> **Note:** The long and chaotic story documented below is specifically regarding the **ROCm driver** and its bugs on Windows.
> 
> **Key Takeaway for Backends & BIOS UMA:**
> - **ROCm-CUDA** is the fastest backend, but it currently has memory allocation bugs. To use it, you **must set UMA to 64GB in BIOS** (96GB UMA is very buggy or impossible to use properly with ROCm).
> - **Vulkan and Vulkan-CUDA** backends **do not** have these bugs. For these backends, you **can safely set UMA to 96GB in BIOS**.
>
> *(Note: Test results have been temporarily removed and will be retested and included later.)*

There are **two independent bugs** that affect HIP memory on Strix Halo. They have different root causes, different effects, and require different fixes. Understanding the distinction is critical for choosing the right BIOS UMA setting.

## Bug 1: `isLargeBar` — allocation size cap (ROCm runtime)

**What it does**: The ROCm runtime hardcodes `isLargeBar = false` for all APUs in HIP mode ([confirmed bug](https://github.com/ROCm/rocm-systems/issues/4077)). Without large-BAR, `hipMalloc` is capped at ~64 GiB (Windows GART 50%-of-RAM limit). Any single allocation request above this cap fails with `hipErrorOutOfMemory`.

**What it does NOT do**: It does not affect *where* data is placed. If an allocation succeeds, it goes to VRAM. The bug only limits the maximum allocation *size*.

**Fix**: Binary patch of `amdhip64_7.dll` — see [Binary patch](#binary-patch-amdhip64_7dll-islargebar-fix) below.

## Bug 2: KV cache spill to shared memory (ROCm + Windows WDDM)

**What it does**: At 96 GB UMA (leaving only 32 GB for Windows), ROCm places the KV cache in slow shared system memory instead of dedicated VRAM — even when VRAM has plenty of space. This causes generation speed to drop dramatically (e.g. 23 tok/s → 9 tok/s). It affects even small 40 GiB models.

**What it does NOT do**: Allocations still *succeed* — the data just lands in the wrong (slow) memory pool.

**Upstream issues**:
- [#18011 — KV cache always dumps into shared memory at 96 GB UMA](https://github.com/ggml-org/llama.cpp/issues/18011)
- [#18159 — UMA detection incorrectly limits available memory on AMD APUs](https://github.com/ggml-org/llama.cpp/issues/18159)

**Root causes**:
1. **ROCm/HIP side**: With 96 GB UMA and only 32 GB left for Windows, GART and paging subsystems are starved, causing ROCm to fall back to shared memory paths.
2. **llama.cpp side**: The UMA detection in `ggml-cuda.cu` checks `prop.integrated > 0` (true for AMD APUs) and overrides `hipMemGetInfo()` with `/proc/meminfo` on Linux, underreporting available memory. A [fix PR #20472](https://github.com/ggml-org/llama.cpp/pull/20472) guards this with `!defined(GGML_USE_HIP)`. On Windows, this code path doesn't run — the bug is purely on the ROCm driver side.

**Fix**: No user-side fix on Windows. The bug is in the WDDM/ROCm driver interaction. Workaround: reduce BIOS UMA to 64 GB. On Linux, TTM kernel parameters can work around this — see [Linux workarounds](#workarounds-for-linux-users).

## How the two bugs interact

| Bug | Affects | Trigger | Fix |
|-----|---------|---------|-----|
| **Bug 1** (isLargeBar) | Max allocation size | Always on (unpatched driver) | Binary patch `amdhip64_7.dll` |
| **Bug 2** (KV spill) | Memory placement speed | 96 GB UMA + low OS RAM | Reduce BIOS UMA to 64 GB |

At 96 GB UMA, both bugs compound: Bug 1 caps allocations at ~64 GiB (fixable with patch), but even after fixing Bug 1, Bug 2 still sends KV cache to shared memory (slow). At 64 GB UMA, Bug 1 is irrelevant (VRAM = 64 GiB = GART cap) and Bug 2 doesn't trigger (OS has 64 GB headroom).

---

## Scenarios: 64 GB vs 96 GB UMA on 128 GB Strix Halo

### Scenario 1: 64 GB GPU + 64 GB OS RAM (recommended, current)

```
128 GB total RAM
├── 64 GB → AMD iGPU VRAM (BIOS UMA)
└── 64 GB → Windows OS + applications
```

| Property | Value |
|----------|-------|
| AMD VRAM | 64 GiB dedicated |
| NVIDIA VRAM | 32 GiB dedicated (PCIe) |
| Combined GPU ceiling | ~96 GiB |
| Windows RAM | 64 GiB (ample headroom) |
| isLargeBar patch needed? | **No** — VRAM (64 GiB) ≤ GART cap (64 GiB) |
| KV cache placement | ✅ Correctly in VRAM |
| KV cache spill bug? | **No** — OS has enough RAM for GART/paging |
| Max model (AMD only) | ~60 GiB (need room for KV cache) |
| Max model (dual GPU) | ~92 GiB split across both GPUs |
| `--no-mmap` safe? | Yes for models ≤ ~60 GiB; risky for larger (double-buffering) |

**Status**: ✅ Working. All benchmarks were collected with this configuration.

### Scenario 2: 96 GB GPU + 32 GB OS RAM (broken, more VRAM)

```
128 GB total RAM
├── 96 GB → AMD iGPU VRAM (BIOS UMA)
└── 32 GB → Windows OS + applications
```

| Property | Value |
|----------|-------|
| AMD VRAM | 96 GiB dedicated |
| NVIDIA VRAM | 32 GiB dedicated (PCIe) |
| Combined GPU ceiling | ~128 GiB (theoretical) |
| Windows RAM | 32 GiB (barely enough) |
| isLargeBar patch needed? | **Yes** — without it, `hipMalloc` caps at ~64 GiB despite 96 GiB VRAM |
| KV cache placement | ❌ Spills to shared memory (slow) |
| KV cache spill bug? | **Yes** — OS RAM too low, GART starved |
| Max model (AMD only) | ~84 GiB (with patch) but KV cache is slow |
| Max model (dual GPU) | ~98 GiB (theoretical) but KV spill kills performance |
| Generation speed | ~60% slower than Scenario 1 on same model |
| System stability | Freezes possible under memory pressure |

**Status**: ❌ Broken. Even small models (~40 GiB) exhibit KV cache spill. The extra 32 GiB of VRAM is unusable in practice because all data landing there performs at shared memory speeds.

### Why Scenario 2 is worse despite more VRAM

The paradox: 96 GB UMA gives 50% more VRAM but delivers ~60% *worse* generation performance. The reason is that Windows needs sufficient OS RAM for:
- GART (Graphics Address Remapping Table) management
- Page file backing for virtual memory
- Desktop Window Manager, drivers, and background processes
- WDDM (Windows Display Driver Model) resource scheduling

With only 32 GB for all of this, Windows memory management degrades, and the ROCm driver falls back to shared memory paths for KV cache — even though dedicated VRAM is available. This is not a "near the limit" issue; it affects even small models well below the 96 GiB ceiling.

### Ideal (future) scenario: 96 GB GPU + 32 GB OS with fixes

If AMD fixes the KV cache spill bug in the ROCm driver and the isLargeBar patch gets merged upstream:

| Property | Value |
|----------|-------|
| AMD VRAM | 96 GiB (all usable) |
| Combined GPU ceiling | ~128 GiB |
| Max model (dual GPU) | ~120 GiB |
| Requirements | Fixed ROCm driver + isLargeBar fix + llama.cpp PR [#20472](https://github.com/ggml-org/llama.cpp/pull/20472) |

This would enable models like MiniMax-M2.5 Q3_K_M (101.76 GiB) to load fully on GPU.

---

## Binary patch: `amdhip64_7.dll` (isLargeBar fix)

> **Note**: This patch is only needed for Scenario 2 (96 GB UMA) or higher UMA settings. At 64 GB UMA (Scenario 1, current), the patch has no effect because VRAM already equals the GART cap.

### Problem

The AMD iGPU reports `isLargeBar: 0` via HIP due to a [confirmed ROCm runtime bug](https://github.com/ROCm/rocm-systems/issues/4077). In `paldevice.cpp`, the runtime hardcodes `info_.largeBar_ = false` for **all** APUs when running in HIP mode, even though Strix Halo has no invisible VRAM heap and should be treated as large-BAR.

Without the fix, `hipMalloc` is capped at ~64 GiB (Windows GART 50%-of-RAM limit), preventing large model loads at higher UMA settings.

### Patch details

The fix is a **single-byte change** in `amdhip64_7.dll` (v10.0.3665.0, SHA256 `0f2b166b...`):

| Property | Value |
|----------|-------|
| File | `C:\Windows\SYSTEM32\amdhip64_7.dll` |
| File offset | `0x003F69BA` |
| Original byte | `0x74` (`JZ` — skip heap check for HIP) |
| Patched byte | `0xEB` (`JMP` — always run heap check) |
| Effect | Forces the `GpuHeapInvisible` size check path, which correctly sets `largeBar_ = true` on APUs with no invisible VRAM |

**Backup**: `amdhip64_7.dll.original` in the project root.
**Patched copy**: `amdhip64_7.dll.patched` in the project root, also placed in `runtime-rocm-cuda/`.

### Code flow (disassembly)

```
003F75B3: CMP [RIP+0xCF359E], SIL       ; check IS_HIP global
003F75BA: JZ  003F75C8                   ; if not HIP → heap check (PATCHED: JMP always)
003F75BC: MOV BYTE [RDI+0x530], SIL      ; HIP path: largeBar_ = false
003F75C3: JMP 003F7649                   ; skip heap check
003F75C8: CMP [R13+0x20], RSI            ; check heapProps[invisible].logicalSize == 0
003F75CC: JNZ 003F7649                   ; has invisible heap → skip
003F75CE: MOV BYTE [RDI+0x530], 1        ; largeBar_ = true ✓
003F75D5: (log "Resizable bar enabled")
```

### Applying the patch

```powershell
# Create patched DLL (run once)
python -c "
data = bytearray(open(r'C:\Windows\SYSTEM32\amdhip64_7.dll','rb').read())
assert data[0x3F69BA] == 0x74, 'Unexpected byte - wrong DLL version'
data[0x3F69BA] = 0xEB
open(r'amdhip64_7.dll.patched','wb').write(data)
"

# Install (requires Administrator)
takeown /F C:\Windows\SYSTEM32\amdhip64_7.dll /A
icacls C:\Windows\SYSTEM32\amdhip64_7.dll /grant Administrators:F
copy amdhip64_7.dll.patched C:\Windows\SYSTEM32\amdhip64_7.dll
```

### Result (tested at 96 GB UMA — Scenario 2)

These results apply to Scenario 2 (96 GB UMA). At 64 GB UMA (Scenario 1), the patch has no observable effect.

| Metric | Before patch | After patch |
|--------|-------------|-------------|
| `isLargeBar` | 0 | 1 |
| Max `hipMalloc` (standalone) | ~64 GiB | ~84 GiB |
| Max `hipMalloc` (with 28 GiB on NVIDIA) | ~36 GiB | ~70 GiB |

> **Note**: AMD driver updates or HIP SDK reinstalls will overwrite the system DLL. Re-apply the patch after updates. The upstream [fix PR](https://github.com/ROCm/rocm-systems/issues/4077) has been approved but not yet merged.

> **Important**: Even with the isLargeBar patch applied, Scenario 2 (96 GB UMA) still suffers from Bug 2 (KV cache spill). The patch lets you allocate >64 GiB, but the data may land in slow shared memory. See Scenarios above.

---

## Potential workarounds for Scenario 2 (96 GB UMA) on Windows

Scenario 2 is currently broken (see above), but several workarounds may help. Use `scripts\diagnose-hip-memory.ps1` to diagnose the HIP memory state and `scripts\test-96gb-uma.ps1` to test models with workarounds applied.

| # | Workaround | Rationale |
|---|-----------|----------|
| 1 | **Large page file (64–128 GB) on NVMe** | With only 32 GB OS RAM, Windows needs virtual memory headroom. A large page file on fast NVMe may prevent the GART starvation that triggers KV cache spill |
| 2 | **Quantized KV cache** (`--cache-type-k q8_0 --cache-type-v q8_0`) | Halves the KV cache footprint — may keep it in VRAM below the spill threshold |
| 3 | **`--flash-attn on`** | Flash attention reduces peak KV memory usage during inference |
| 4 | **`--cache-ram 0`** | Disables the prompt cache, freeing VRAM for model + KV |
| 5 | **Kill all GPU-using processes** | With 32 GB OS RAM, DWM, browsers, etc. consume precious shared memory |
| 6 | **Disable Hyper-V** | Hyper-V reserves a memory partition that compounds the pressure |

**What WON'T work on Windows:**
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — on Windows+HIP, `hipMallocManaged` falls back to `hipMalloc` silently (and is broken on Strix Halo anyway)
- The llama.cpp UMA fix (PR [#20472](https://github.com/ggml-org/llama.cpp/pull/20472)) — it's `#if defined(__linux__)` only, no effect on Windows
- `--no-mmap` with large models — double-buffering (93 GiB model + 96 GiB VRAM copy ≈ 189 GiB) exceeds 128 GB physical RAM

### Testing procedure

```powershell
# Step 1: Set BIOS UMA to 96 GB, set page file to 128 GB on NVMe, reboot

# Step 2: Run memory diagnostics
.\scripts\diagnose-hip-memory.ps1
# Look for: hipMemGetInfo total, isLargeBar, max hipMalloc ceiling

# Step 3: Test small model with quantized KV cache
.\scripts\test-96gb-uma.ps1 -ModelPath "C:\path\to\Qwen3.5-27B.Q4_K_M.gguf" -Context 4096
# Look for: "ROCm0 KV buffer" (good) vs "ROCm_Host KV buffer" (bad = spill)

# Step 4: Test same model WITHOUT quantized KV (baseline comparison)
.\scripts\test-96gb-uma.ps1 -ModelPath "C:\path\to\Qwen3.5-27B.Q4_K_M.gguf" -Context 4096 -NoKVQuant

# Step 5: Test with larger context
.\scripts\test-96gb-uma.ps1 -ModelPath "C:\path\to\Qwen3.5-27B.Q4_K_M.gguf" -Context 32768

# Step 6: Test large model (60+ GiB)
.\scripts\test-96gb-uma.ps1 -ModelPath "C:\path\to\large-model.gguf" -Context 4096
```

---

## Workarounds for Linux users

- Use 64 GB UMA + TTM kernel parameters for additional GPU-accessible memory:
  ```
  amdgpu.gttsize=98304 ttm.pages_limit=25000000 ttm.page_pool_size=12000000
  ```
- Add `amdgpu.cwsr_enable=0` to prevent ROCm crashes
- Use kernel 6.18.4+ with latest ROCm for memory management fixes
- Alternatively: set VRAM as low as possible (512 MB) in BIOS and use `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON` — this uses system RAM instead of VRAM/TTM. See [#18159 comment by Djip007](https://github.com/ggml-org/llama.cpp/issues/18159#issuecomment-2621025851).

---

## Shared memory ceiling (AMD iGPU + NVIDIA dGPU)

With Scenario 1 (**64 GB UMA**), the AMD iGPU has 64 GiB of dedicated VRAM carved from system RAM. The NVIDIA RTX 5090 has its own 32 GB dedicated VRAM.

**Total GPU memory: 64 GiB (AMD) + 32 GiB (NVIDIA) = 96 GiB** (Scenario 1)

Because AMD iGPU VRAM is carved from system RAM, NVIDIA PCIe memory-mapped allocations can still reduce the effective pool:

| NVIDIA allocation | Max HIP allocation | Combined total |
|-------------------|-------------------|----------------|
| 0 GiB | ~64 GiB | ~64 GiB |
| 28 GiB | ~64 GiB | ~92 GiB |
| 32 GiB | ~64 GiB | ~96 GiB |

> **Scenario 2 (96 GB UMA)** would theoretically raise the AMD allocation to ~84 GiB (with isLargeBar patch) for a ~128 GiB combined ceiling, but the KV cache spill bug (Bug 2) makes it unusable — see Scenarios above.

### Implications for model loading

- Models up to ~92 GiB can be split across both GPUs with `--tensor-split`
- The MiniMax-M2.5 Q3_K_M model (101.76 GiB) cannot fit entirely in GPU memory — some layers must remain on CPU
- The NVIDIA RTX 5090 uses dedicated 32 GB VRAM (not shared), but its PCIe memory-mapped allocations still consume address space visible to the unified memory controller
- `GGML_CUDA_NO_PINNED=1` is recommended to avoid pinned (host) memory allocation failures

---

## Known issue: `hipMallocManaged` is broken on Strix Halo

`hipMallocManaged` (unified/managed memory) is completely non-functional on this hardware. After the first successful call, all subsequent calls return `hipErrorOutOfMemory` regardless of size — even 1 byte.

**Impact**: The `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` environment variable must **not** be set, as it causes `ggml-hip.dll` to use `hipMallocManaged` instead of `hipMalloc`. With it enabled, the HIP backend fails to allocate any tensors after the first one.

This appears to be a Strix Halo / gfx1151-specific driver bug. Regular `hipMalloc` works correctly up to the VRAM ceiling.

> **Note on Linux**: On Linux, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON` uses system RAM allocation (not VRAM/TTM), so it may work but will be slower. If using this mode, set VRAM as low as possible in BIOS and do not modify TTM parameters. See [#18159 comment by Djip007](https://github.com/ggml-org/llama.cpp/issues/18159#issuecomment-2621025851) for details.
