# Llama Server on Windows (RTX 5090 + AMD Strix Halo iGPU)

> **Note:** This project was primarily developed and optimized for an **AMD Strix Halo 395+ with 128GB RAM** paired with one dedicated GPU card, but it can be easily modified or adapted to run on any Windows machine with 2 or more GPUs.

Dual-backend llama.cpp server using **CUDA** (NVIDIA RTX 5090) and **ROCm/HIP** (AMD Radeon 8060S iGPU) simultaneously, with Open WebUI as a web frontend. Also supports **Vulkan** as an alternative backend that can drive both GPUs without ROCm/HIP.

## Project snapshot

- Supports dual-backend testing for AMD + NVIDIA on Windows.
- Features `rocm-cuda`, `vulkan`, and `vulkan-cuda` backend modes.
- Includes memory diagnostic scripts and patching utilities for ROCm.

## Hardware

| Component | Details |
|-----------|---------|
| CPU | AMD Ryzen AI MAX+ 395 (32 threads) |
| RAM | 128 GB unified (64 GB GPU + 64 GB OS) |
| GPU 1 | AMD Radeon 8060S iGPU (RDNA 3.5, gfx1151, 64 GiB VRAM via BIOS UMA, 512-bit bus) |
| GPU 2 | NVIDIA GeForce RTX 5090 (32 GB VRAM, compute 12.0, PCIe 4.0 x4) |




## Backend options

| Backend | AMD GPU | NVIDIA GPU | Runtime dir | Build requirement |
|---------|---------|------------|-------------|-------------------|
| `rocm-cuda` | ROCm/HIP | CUDA | `runtime-rocm-cuda/` | HIP SDK 7.1.1 + CUDA Toolkit |   
| `vulkan` | Vulkan | — | `runtime-vulkan/` | LunarG Vulkan SDK |
| `vulkan-vulkan` | Vulkan | Vulkan | `runtime-vulkan/` | LunarG Vulkan SDK |   
| `vulkan-cuda` | Vulkan | CUDA | `runtime-vulkan-cuda/` | LunarG Vulkan SDK + CUDA Toolkit |

**ROCm+CUDA** is the current default (best performance, requires HIP SDK + binary patch).
**Vulkan** is a simpler alternative — no HIP SDK needed, no `amdhip64_7.dll` patch, and a single `ggml-vulkan.dll` can drive both AMD and NVIDIA GPUs. May also avoid the ROCm memory bugs.



## Recommended Modes on Windows (AMD Strix Halo + Dedicated GPU)

For a dual-GPU system based on an AMD Strix Halo (e.g., AMD Strix Halo 395) combined with an NVIDIA dedicated GPU (e.g., RTX 5090 32GB), base your backend Mode and BIOS settings on the total memory footprint of the model and context:

As of 2026-05-29, this machine has also successfully loaded and served requests with **`rocm-cuda` at 96 GB UMA** using the patched HIP runtime and current CUDA/ROCm stack. Treat that as a verified observation for this setup, not a universal guarantee. **64 GB UMA** remains the conservative recommendation for ROCm on Windows until more benchmark and stability data is collected.

**Example for AMD Strix Halo + RTX 5090 32GB**

* **Model + Context < 126-128GB**
  If the model and its context can comfortably fit within ~128GB (94GB AMD + 32GB NVIDIA), choose the **`rocm-cuda`** mode for the best performance.
  * **BIOS Setting:** Set UMA GPU size to **64GB**.


* **Model + Context > 126-128GB (Up to 160GB)**
    If you are running exceptionally large models or huge context windows that exceed 96GB (on AMD Strix Halo), **`vulkan-cuda`** or **`vulkan-vulkan`** are still the safer choices. Historically, ROCm on Windows topped out near ~64GB contiguous UMA allocation and also showed poor 96 GB UMA behavior on this hardware. Current testing shows that **`rocm-cuda` can now start and serve requests at 96 GB UMA on this machine**, but that path should still be benchmarked and monitored for long-run stability before treating it as the default recommendation. Vulkan still circumvents the older ROCm limitations more reliably.
  * **BIOS Setting:** Set UMA GPU size to **96GB**.



## Runtime

Each backend has its own runtime directory containing `llama-server.exe` and backend DLLs.

**ROCm+CUDA** (`runtime-rocm-cuda/`): `ggml-hip.dll` + `ggml-cuda.dll` — devices: `ROCm0`, `CUDA0`
**Vulkan** (`runtime-vulkan/`): `ggml-vulkan.dll` — devices: `Vulkan0`, `Vulkan1`
**Vulkan+CUDA** (`runtime-vulkan-cuda/`): `ggml-vulkan.dll` + `ggml-cuda.dll` — devices: `Vulkan0`, `CUDA0`

### Modes

| Mode | Devices | Runtime |
|------|---------|---------|
| `rocm` | AMD single GPU (ROCm) | `runtime-rocm-cuda/` |
| `cuda` | NVIDIA single GPU (CUDA) | `runtime-rocm-cuda/` |
| `vulkan` | Single GPU via Vulkan (first detected) | `runtime-vulkan/` |       
| `rocm-cuda` | AMD ROCm + NVIDIA CUDA dual GPU | `runtime-rocm-cuda/` |

### Fast Switching (Testing Multiple Backends)
You can compile multiple backends at once and keep them indefinitely:
```powershell
.\scripts\setup-llama.ps1 -Backend rocm-cuda
.\scripts\setup-llama.ps1 -Backend vulkan
.\scripts\setup-llama.ps1 -Backend vulkan-cuda
```
The script will isolate them into separate `runtime-<backend>` folders without overwriting previous runs. When launching the server, you can switch between frameworks seamlessly by changing the `-Mode` flag for `start-llama-server.ps1` (or your model-specific script) to the corresponding `Mode` in the table above.

- **[Benchmarks](doc/benchmarks.md)** — Performance results for some models across single/dual GPU configurations
- **[ROCm Bugs & Workarounds](doc/bugs.md)** — Two Strix Halo memory bugs, BIOS UMA scenarios, binary patch, workarounds

## Known ROCm bugs (summary)

There are **two independent bugs** affecting HIP memory on Strix Halo. Full details in [doc/bugs.md](doc/bugs.md).

| Bug | What it does | Fix |
|-----|-------------|-----|
| **Bug 1: `isLargeBar`** | Caps `hipMalloc` at ~64 GiB ([upstream](https://github.com/ROCm/rocm-systems/issues/4077)) | Binary patch `amdhip64_7.dll` — only needed at >64 GB UMA |
| **Bug 2: KV cache spill** | Places KV cache in slow shared memory at 96 GB UMA ([#18011](https://github.com/ggml-org/llama.cpp/issues/18011)) | No user-side fix on Windows — use 64 GB UMA instead |

**Recommended BIOS setting**: 64 GB UMA (64 GB GPU + 64 GB OS). At this setting Bug 1 is irrelevant and Bug 2 doesn't trigger. Setting 96 GB UMA is broken — see [doc/bugs.md](doc/bugs.md) for details.

**Do NOT set** `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — `hipMallocManaged` is broken on Strix Halo (gfx1151). See [doc/bugs.md](doc/bugs.md#known-issue-hipmallocmanaged-is-broken-on-strix-halo).

## Prerequisites

**Common** (all backends):
- Visual Studio 2022 Build Tools with `Desktop development with C++`
- CMake, Git, Ninja (in PATH)
- Docker Desktop (for Open WebUI)

**ROCm+CUDA** (default):
- NVIDIA driver (latest)
- AMD HIP SDK 7.1.1 (includes PRO driver): `AMD-Software-PRO-Edition-26.Q1-Win11-For-HIP.exe`
- CUDA Toolkit 13.2

**Vulkan**:
- NVIDIA driver (latest — includes Vulkan ICD)
- AMD driver (already installed with system)
- [LunarG Vulkan SDK](https://vulkan.lunarg.com/) (provides `glslc` shader compiler needed at build time)

**Vulkan+CUDA** (hybrid):
- All Vulkan prerequisites plus CUDA Toolkit 13.2

> **MSVC toolset note**: CUDA 13.2 is incompatible with MSVC v14.50+ (VS 18 / v145 toolset) — `cudafe++` crashes with `ACCESS_VIOLATION`. The build script auto-detects this and falls back to an older toolset (e.g. 14.44). If no compatible toolset is found, install the **MSVC v143** component via the Visual Studio Installer (Individual Components → "MSVC v143 - VS 2022 C++ x64/x86 build tools").

## Usage

### Build from source

```powershell
# ROCm+CUDA (default)
.\scripts\setup-llama.ps1

# Vulkan only
.\scripts\setup-llama.ps1 -Backend vulkan

# Vulkan + CUDA hybrid
.\scripts\setup-llama.ps1 -Backend vulkan-cuda
```

### Discover devices

```powershell
.\scripts\list-devices.ps1                    # ROCm+CUDA runtime
.\scripts\list-devices.ps1 -Backend vulkan    # Vulkan runtime
```

### Run server (dual GPU)

```powershell
# ROCm + CUDA (default)
.\scripts\start-llama-server.ps1 -Mode rocm-cuda -ExtraArgs @(
    "-m", "path\to\model.gguf",
    "-c", "8192",
    "-ngl", "99",
    "--tensor-split", "1,1"
)

# Vulkan dual (both GPUs via Vulkan)
.\scripts\start-llama-server.ps1 -Mode vulkan-vulkan -ExtraArgs @(
    "-m", "path\to\model.gguf",
    "-c", "8192",
    "-ngl", "99",
    "--tensor-split", "1,1"
)

# Vulkan AMD + CUDA NVIDIA
.\scripts\start-llama-server.ps1 -Mode vulkan-cuda -ExtraArgs @(
    "-m", "path\to\model.gguf",
    "-c", "8192",
    "-ngl", "99",
    "--tensor-split", "1,1"
)
```

### Run server (single GPU)

```powershell
# CUDA only (NVIDIA)
.\scripts\start-llama-server.ps1 -Mode cuda -ExtraArgs @(
    "-m", "path\to\model.gguf",
    "-c", "8192",
    "-ngl", "99"
)

# ROCm only (AMD)
.\scripts\start-llama-server.ps1 -Mode rocm -ExtraArgs @(
    "-m", "path\to\model.gguf",
    "-c", "8192",
    "-ngl", "99"
)

# Vulkan only (first Vulkan device)
.\scripts\start-llama-server.ps1 -Mode vulkan -ExtraArgs @(
    "-m", "path\to\model.gguf",
    "-c", "8192",
    "-ngl", "99"
)
```

### Model profiles

Profile scripts define model-specific llama-server arguments and accept `-Mode` to switch GPU backends:

```powershell
.\scripts\start-minimax-2.5-mxfp4.ps1                          # vulkan-vulkan (default)
.\scripts\start-minimax-2.5-mxfp4.ps1 -Mode rocm-cuda          # ROCm+CUDA

.\scripts\start-qwen35-27b-opus-q4km.ps1                        # vulkan-cuda (default)
.\scripts\start-qwen35-27b-opus-q4km.ps1 -Mode cuda             # NVIDIA CUDA only

.\scripts\start-qwen122b-q6k.ps1                                # rocm-cuda (default)
.\scripts\start-qwen122b-q6k.ps1 -Mode vulkan-vulkan            # Vulkan dual
```

Each profile script is a thin wrapper that calls `start-llama-server.ps1` with `-Mode` and `-ExtraArgs` containing native llama-server flags (`-m`, `-c`, `-ngl`, `--tensor-split`, etc.).

### Diagnose HIP memory

Run after changing BIOS UMA allocation to verify HIP reports correct memory and allocations work:

```powershell
.\scripts\diagnose-hip-memory.ps1
```

Reports: Windows physical RAM, page file, `hipMemGetInfo`, `isLargeBar`, `hipMalloc` ceiling, `hipMallocManaged` status, and KV cache simulation.

### Test 96 GB UMA workarounds

```powershell
# ROCm-only with quantized KV cache
.\scripts\test-96gb-uma.ps1 -ModelPath "path\to\model.gguf" -Context 4096

# Baseline comparison (no KV quantization)
.\scripts\test-96gb-uma.ps1 -ModelPath "path\to\model.gguf" -Context 4096 -NoKVQuant
```

### Open WebUI frontend

```powershell
.\scripts\start-open-webui.ps1 -ApiBase "http://host.docker.internal:8080/v1"
```

Then open:
- Open WebUI: http://localhost:3000
- Llama server UI: http://localhost:8080

## Quick health check

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

Expected once loaded:

```json
{"status":"ok"}
```

## TODO

- [ ] **ROCm nightlies (experimental)**: Build llama.cpp against [TheRock nightly ROCm builds](https://therock-nightly-tarball.s3.amazonaws.com/index.html) for `gfx1151`. A newer ROCm runtime might handle memory placement better and fix Bug 2 (KV cache spill). Suggested by slewsys in [#18011](https://github.com/ggml-org/llama.cpp/issues/18011) — see [build guide](https://gist.github.com/slewsys/06f972dc796109c0777c64b2d4b2c924). Untested, requires full rebuild.
- [ ] **RPC dual-GPU mode**: Run separate `rpc-server.exe` processes per GPU backend (one CUDA, one ROCm) on different ports, then connect `llama-server` via `--rpc 127.0.0.1:50051,127.0.0.1:50052`. Gives each backend its own address space. However, **unlikely to help on this system**: the ~96 GiB ceiling (64 AMD + 32 NVIDIA) is a physical hardware limit, not an address space limit. Bug 1 (isLargeBar) and Bug 2 (KV spill) are system-wide driver issues that affect any process using ROCm. RPC would not increase available VRAM or fix memory placement. Low priority.
- [x] **Vulkan backend support**: Added Vulkan as alternative to ROCm/HIP. Build with `setup-llama.ps1 -Backend vulkan` or `vulkan-cuda`. A single `ggml-vulkan.dll` drives both AMD and NVIDIA GPUs. No HIP SDK or binary patch needed. All scripts accept `-Mode vulkan-vulkan` / `vulkan-cuda`. MSVC toolset auto-detection handles CUDA 13.2 + VS 18 incompatibility.
- [ ] **Open WebUI**: Set up Open WebUI via Docker Desktop connecting to `llama-server`.
