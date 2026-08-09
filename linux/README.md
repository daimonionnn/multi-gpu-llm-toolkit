# Linux — Radeon AI PRO R9700 + RTX PRO 6000

Bash port of the multi-GPU llama.cpp setup, targeting the **`dual-linux`** rig:
an AMD Radeon AI PRO R9700 (32 GB, RDNA 4) alongside an NVIDIA RTX PRO 6000
(96 GB, Blackwell) on an Intel platform.

> **Status: in progress.** The build and launch scripts are being ported from
> the [Windows implementation](../windows/README.md), which is complete and
> working. Nothing in this directory is benchmarked yet.

## How this differs from the Windows side

The Windows rig pairs an APU with a discrete card. This one has two discrete
cards, which removes most of the hard parts and adds a couple of new ones.

**Does not apply here:**

- No BIOS UMA split — there is no shared memory pool to size.
- No `amdhip64_7.dll` binary patch. The `isLargeBar` cap is an APU-mode bug; a
  discrete card reports large-BAR normally.
- No KV-cache spill to shared system memory. There is no shared pool to spill
  into; the R9700's 32 GB is dedicated VRAM.
- No WDDM. Memory management goes through `amdgpu`/TTM instead, which is where
  the Linux workarounds mentioned in [../doc/rocm-bugs.md](../doc/rocm-bugs.md)
  come from.

**New concerns instead:**

- **gfx1201 support.** The R9700 is RDNA 4. Confirm ROCm builds for it natively
  rather than needing `HSA_OVERRIDE_GFX_VERSION`, and that llama.cpp's HIP
  backend targets it. Verify with `rocminfo` before building anything.
- **PCIe topology.** Two real cards means the split between them is bandwidth-
  sensitive in a way an on-die iGPU never was. `--tensor-split` will need
  retuning from scratch; the Windows ratios do not carry over.
- **Asymmetric VRAM.** 96 GB and 32 GB is a 3:1 ratio. An even
  `--tensor-split 1,1` wastes most of the RTX PRO 6000 — expect something closer
  to `1,3` as a starting point, then tune.
- **Driver coexistence.** `amdgpu` and the proprietary NVIDIA driver must be
  loaded at once, with both ROCm and CUDA initialised inside a single process.

## Planned scripts

Mirroring the Windows script names so the two implementations stay comparable:

| Script                      | Purpose                                               | Windows equivalent           |
|-----------------------------|-------------------------------------------------------|------------------------------|
| `list-devices.sh`           | Enumerate GPUs and their backend device names         | `list-devices.ps1`           |
| `setup-llama.sh`            | Build llama.cpp per backend into `runtime-<backend>/` | `setup-llama.ps1`            |
| `start-llama-server.sh`     | Launch the server in a chosen mode                    | `start-llama-server.ps1`     |
| `run-llama-bench.sh`        | Raw hardware benchmark via `llama-bench`              | `run-llama-bench.ps1`        |
| `benchmark-loaded-model.sh` | End-to-end HTTP benchmark against a running server    | `benchmark-loaded-model.ps1` |
| `start-open-webui.sh`       | Open WebUI frontend via Docker                        | `start-open-webui.ps1`       |

Backend modes are the same as on Windows — `rocm-cuda`, `vulkan`,
`vulkan-vulkan`, `vulkan-cuda` — with `.so` libraries in place of `.dll`:

| Backend       | Libraries                               | Device names       |
|---------------|-----------------------------------------|--------------------|
| `rocm-cuda`   | `libggml-hip.so` + `libggml-cuda.so`    | `ROCm0`, `CUDA0`   |
| `vulkan`      | `libggml-vulkan.so`                     | `Vulkan0`          |
| `vulkan-cuda` | `libggml-vulkan.so` + `libggml-cuda.so` | `Vulkan0`, `CUDA0` |

## Prerequisites

**Common:** `build-essential`, `cmake`, `ninja-build`, `git`, and Docker if you
want Open WebUI.

**ROCm+CUDA:** ROCm (version supporting gfx1201) and the CUDA Toolkit, plus the
proprietary NVIDIA driver.

**Vulkan:** Vulkan loader and headers, `glslc` (shader compiler, needed at build
time), the Mesa RADV or AMDVLK ICD for the AMD card, and the NVIDIA driver's
Vulkan ICD.

Exact package names and versions will be pinned here once the setup is verified
on the actual machine.

## First step

Before porting anything, characterise the hardware:

```bash
rocminfo | grep -E 'Name:|gfx'
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
```

Record the results in [../doc/systems.md](../doc/systems.md), which currently
has placeholders for this rig.
