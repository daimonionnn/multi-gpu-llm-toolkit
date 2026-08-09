# Linux — Radeon AI PRO R9700 + RTX PRO 6000

Bash port of the multi-GPU llama.cpp setup, targeting the **`dual-linux`** rig:
an AMD Radeon AI PRO R9700 (32 GB, RDNA 4, gfx1201) alongside an NVIDIA
RTX PRO 6000 Blackwell (96 GB, sm_120) on Ubuntu 26.04.

> **Status:** all scripts ported from the [Windows implementation](../windows/README.md)
> and their detection, argument handling and error paths exercised on the real
> machine. **No build has been run yet** — `hipcc` and `glslc` are still missing
> — so nothing is benchmarked and the build path itself is unverified.

## Quick start

```bash
cd linux/scripts
./install-prereqs.sh --backend all      # apt packages (does not touch GPU drivers)
./list-devices.sh                       # what the system exposes
./setup-llama.sh --backend rocm-cuda    # build into ../runtime-rocm-cuda/
./start-llama-server.sh --mode rocm-cuda -- -m /path/model.gguf -c 8192 -ngl 99
```

Everything after `--` goes to `llama-server` untouched — the equivalent of
`-ExtraArgs` in the PowerShell scripts.

## Scripts

| Script                      | Purpose                                            | Windows equivalent            |
|-----------------------------|----------------------------------------------------|-------------------------------|
| `common.sh`                 | Shared helpers (sourced, not run)                  | inline functions              |
| `install-prereqs.sh`        | apt packages for the chosen backend                | `install-prereqs.ps1`         |
| `list-devices.sh`           | GPUs per vendor, plus llama.cpp's own device list  | `list-devices.ps1`            |
| `setup-llama.sh`            | Build llama.cpp into `runtime-<backend>/`          | `setup-llama.ps1`             |
| `start-llama-server.sh`     | Launch the server in a chosen mode                 | `start-llama-server.ps1`      |
| `run-llama-bench.sh`        | Raw hardware benchmark via `llama-bench`           | `run-llama-bench.ps1`         |
| `benchmark-loaded-model.sh` | End-to-end HTTP benchmark against a running server | `benchmark-loaded-model.ps1`  |
| `start-open-webui.sh`       | Open WebUI frontend via Docker                     | `start-open-webui.ps1`        |
| `start-model-template.sh`   | Model profile template — copy per model            | `start-qwen122b-q6k.ps1` etc. |

Backend modes match Windows — `rocm-cuda`, `vulkan`, `vulkan-vulkan`,
`vulkan-cuda` — with `.so` backends in place of `.dll`:

| Backend       | Libraries                               | Device names       |
|---------------|-----------------------------------------|--------------------|
| `rocm-cuda`   | `libggml-hip.so` + `libggml-cuda.so`    | `ROCm0`, `CUDA0`   |
| `vulkan`      | `libggml-vulkan.so`                     | `Vulkan*`          |
| `vulkan-cuda` | `libggml-vulkan.so` + `libggml-cuda.so` | `Vulkan*`, `CUDA0` |

## What is different from Windows

Two discrete cards instead of an APU plus a card. That removes the hard parts of
the Windows setup and adds different ones.

**Gone:** no BIOS UMA split, no `amdhip64_7.dll` binary patch (the `isLargeBar`
cap is an APU-mode bug), no KV-cache spill into shared memory (there is no
shared pool), no WDDM. See [../doc/rocm-bugs.md](../doc/rocm-bugs.md) for which
bugs can occur on which hardware.

**New, and all four hit this rig in practice:**

- **The `nvcc` in `PATH` is too old for the GPU.** `/usr/bin/nvcc` is CUDA 12.4,
  which stops at `compute_90`; the RTX PRO 6000 is `sm_120`. CUDA 13.1 is
  installed at `/usr/local/cuda-13.1` but is not first in `PATH`, so a naive
  build fails with "unsupported gpu architecture". `setup-llama.sh` reads the
  GPU's compute capability from `nvidia-smi`, scans the installed toolkits and
  pins one that supports it. This is the direct analogue of the MSVC toolset
  workaround in `setup-llama.ps1`.
- **ROCm is not in `/opt/rocm`.** Ubuntu 26.04 ships it as distro packages
  spread across `/usr`, with the CMake config under
  `/usr/lib/x86_64-linux-gnu/cmake/hip`. The scripts probe `hipconfig`,
  `$ROCM_PATH`, `/opt/rocm` and `/usr` in that order.
- **Four Vulkan devices, not two.** Alongside RADV and the NVIDIA ICD, the
  loader also exposes the Intel Arrow Lake iGPU and the llvmpipe software
  rasteriser. Picking `Vulkan0`/`Vulkan1` by index — which is what the Windows
  script does, correctly, on a machine with exactly two — would select the
  iGPU here. `start-llama-server.sh` matches devices by description instead and
  reports which ones it ignored.
- **`--tensor-split` does not carry over.** 96 GB vs 32 GB is roughly 3:1, so an
  even `1,1` caps the model at 64 GB and idles two thirds of the NVIDIA card.
  Device order is AMD first, so `1,3` is the starting point. Untuned.

### One deliberate behavioural difference

`start-llama-server.ps1` always sets `GGML_CUDA_NO_PINNED=1` for ROCm/CUDA
modes. That was a WDDM workaround; on Linux pinned host memory works normally
and disabling it costs transfer bandwidth. Here it is **off by default** and
available as `--no-pinned` if a problem shows up.

## Prerequisites

`install-prereqs.sh` handles the apt side: `build-essential`, `cmake`,
`ninja-build`, `git`, `curl`, `jq`, `libcurl4-openssl-dev`, plus per backend
`hipcc`/`libhipblas-dev`/`librocblas-dev` for ROCm and
`glslc`/`libvulkan-dev`/`vulkan-tools` for Vulkan.

It deliberately does **not** install the CUDA Toolkit or any GPU driver. The
required CUDA version depends on the card, and distro packages are routinely too
old for a current-generation GPU — as they are on this rig. Install a toolkit
from NVIDIA if `install-prereqs.sh` warns that none supports your `sm_` target.

## What has actually been verified

Run on the real machine: GPU/compute-capability detection, CUDA toolkit
selection (correctly prefers 13.1 over the 12.4 in `PATH`), gfx target detection
(`gfx1201`, natively supported — no `HSA_OVERRIDE_GFX_VERSION`), ROCm layout
probing, runtime-directory resolution, device selection for all six modes
against a simulated four-device Vulkan list, the HTTP benchmark end-to-end
against a mock server including concurrent-request aggregation, and every
argument-validation and failure path.

Not verified: the actual llama.cpp build for any backend, and therefore every
number in [../doc/benchmarks.md](../doc/benchmarks.md).
