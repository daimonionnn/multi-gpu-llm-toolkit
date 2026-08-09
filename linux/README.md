# Linux — Radeon AI PRO R9700 + RTX PRO 6000

Bash port of the multi-GPU llama.cpp setup, targeting the **`dual-linux`** rig:
an AMD Radeon AI PRO R9700 (32 GB, RDNA 4, gfx1201) alongside an NVIDIA
RTX PRO 6000 Blackwell (96 GB, sm_120) on Ubuntu 26.04.

> **Status:** all scripts ported from the [Windows implementation](../windows/README.md)
> and exercised on the real machine. **All four backends build**, and
> `rocm-cuda` enumerates ROCm and CUDA together in a single process. A first
> benchmark matrix across all seven single- and dual-GPU configurations is in
> [../doc/benchmarks.md](../doc/benchmarks.md).

## Quick start

```bash
cd linux/scripts
./install-prereqs.sh --backend all   # apt packages (does not touch GPU drivers)
./list-devices.sh                    # what the system exposes
./setup-llama.sh --backend rocm-cuda # build into ../runtime-rocm-cuda/
./start-llama-server.sh --mode rocm-cuda -- -m /path/model.gguf -c 8192 -ngl 99
```

`rocm-cuda` needs a CUDA Toolkit newer than the one Ubuntu 26.04 ships — see
[CUDA on Ubuntu 26.04](#cuda-on-ubuntu-2604) below.

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
| `start-deepseek-mxfp4.sh`   | DeepSeek V4 Flash, MXFP4 (lossless reference): expert-offload dual + RAM, 256k default | — |
| `start-deepseek-mxfp4-cuda-only.sh` | The same, NVIDIA-only — fault-immune profile the systemd fallback service runs | — |
| `start-deepseek-nvidia.sh`  | DeepSeek V4 Flash entirely on the NVIDIA card (antirez IQ2XXS+Q8-attn/128k; `--iq2m`, `--200k`, `--iq3*`) | — |
| `start-deepseek-nvidia-amd.sh` | DeepSeek V4 Flash entirely in VRAM across both cards (Q2_K_XL, 200k) | — |
| `start-step37.sh`           | Step-3.7-Flash Q4_K_S — 104 GB across both cards' VRAM, the true dual-GPU case | — |
| `start-gptoss.sh`           | gpt-oss-120b MXFP4, CUDA-only — fastest model on the rig (258 t/s) | — |
| `build-cuda12-container.sh` | CUDA backend built with CUDA 12.8 in Docker; merges into a dual runtime | — |

Backend modes match Windows — with `.so` backends in place of `.dll`, plus one
addition. `setup-llama.sh` accepts a **`rocm`** backend that the PowerShell
version has no equivalent for: on Windows every HIP build is bundled with CUDA,
which leaves no way to use the AMD card when the CUDA toolchain is unusable.
That is exactly the situation on this rig.

| Backend       | Libraries                               | Device names       | Builds here |
|---------------|-----------------------------------------|--------------------|-------------|
| `rocm`        | `libggml-hip.so`                        | `ROCm0`            | Yes         |
| `vulkan`      | `libggml-vulkan.so`                     | `Vulkan*`          | Yes         |
| `rocm-cuda`   | `libggml-hip.so` + `libggml-cuda.so`    | `ROCm0`, `CUDA0`   | Yes         |
| `vulkan-cuda` | `libggml-vulkan.so` + `libggml-cuda.so` | `Vulkan*`, `CUDA0` | Yes         |

Launch modes are unchanged: `rocm`, `cuda`, `vulkan`, `rocm-cuda`,
`vulkan-vulkan`, `vulkan-cuda`. `vulkan-vulkan` uses the `vulkan` runtime, so
building `vulkan` is enough for dual-GPU operation.

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

## Web UI

Upstream ships the llama-server browser UI as a separate release asset since
`b10331`. Fetch it once and every launched server serves it at its root URL:

```bash
mkdir -p ../webui
gh release download b10331 --repo ggml-org/llama.cpp --pattern "llama-b10331-ui.tar.gz" -O - | tar xz -C ../webui --strip-components=1
```

`start-llama-server.sh` picks `linux/webui/` up automatically (skipped if you
pass your own `--path`). Without it the API still works but `/` returns 415.

## Prerequisites

`install-prereqs.sh` handles the apt side: `build-essential`, `cmake`,
`ninja-build`, `git`, `curl`, `jq`, `libcurl4-openssl-dev`, plus per backend
`hipcc`/`libhipblas-dev`/`librocblas-dev` for ROCm and
`glslc`/`libvulkan-dev`/`vulkan-tools`/`spirv-headers`/`spirv-tools`/`glslang-dev`
for Vulkan. `spirv-headers` is not optional — llama.cpp's shader generation does
`find_package(SPIRV-Headers)`, and the Vulkan configure fails without it even
though `glslc` is present.

It deliberately does **not** install the CUDA Toolkit or any GPU driver. The
required CUDA version depends on the card, and distro packages are routinely too
old for a current-generation GPU — as they are on this rig. Install a toolkit
from NVIDIA if `install-prereqs.sh` warns that none supports your `sm_` target.

## CUDA on Ubuntu 26.04

The distro's CUDA 13.1 — the newest in Ubuntu's own repos, and the only one
there that targets `sm_120` — **cannot compile against glibc 2.43**. Conflicting
`rsqrt`/`rsqrtf` declarations kill every CUDA compilation, down to an empty
kernel, so `rocm-cuda` and `vulkan-cuda` will not build with it.

The fix is CUDA 13.3 from NVIDIA's apt repository, which is what this rig now
uses. Full analysis, the install commands, and four workarounds that do *not*
work: **[../doc/cuda-glibc-243.md](../doc/cuda-glibc-243.md)**.

`setup-llama.sh` finds the newest toolkit supporting the GPU on its own, so no
`PATH` changes are needed after installing it.

## What has actually been verified

Built and run on the real machine:

- **All four backends build** — `rocm`, `vulkan`, `rocm-cuda`, `vulkan-cuda`.
- **Both vendors initialise in one process.** The `rocm-cuda` runtime reports
  `CUDA0: NVIDIA RTX PRO 6000` and `ROCm0: AMD Radeon AI PRO R9700` together,
  which is the whole point of the project and had not been shown on Linux
  before.
- **`vulkan` enumerates** all three usable Vulkan devices; **`rocm`** reports
  `ROCm0: AMD Radeon AI PRO R9700 (32624 MiB)`.
- **Device selection against the real device list.** llama.cpp orders them
  `Vulkan0` = Intel iGPU, `Vulkan1` = NVIDIA, `Vulkan2` = AMD — so index-based
  selection really would have picked the iGPU. `vulkan-vulkan` correctly
  resolves to `Vulkan2,Vulkan1` and reports the ignored iGPU.
- CUDA toolkit selection (prefers 13.1 over the 12.4 in `PATH`), gfx target
  detection (`gfx1201`, no `HSA_OVERRIDE_GFX_VERSION` needed), ROCm layout
  probing, runtime-directory resolution.
- The HTTP benchmark end-to-end against a mock server, including
  concurrent-request aggregation, plus every argument and failure path.

Three bugs in the port itself were found this way and fixed: `GGML_BACKEND_PATH`
was being set to a directory (ggml wants a file, and logged a load error for
each backend), `ROCM_PATH` was being exported as `/usr` on the distro layout
(which makes clang look for device bitcode in `/usr/amdgcn/bitcode` and fail the
compiler check), and `spirv-headers` was missing from the prerequisites.

All seven single- and dual-GPU configurations were then benchmarked with a
21 GB model — results and analysis in [../doc/benchmarks.md](../doc/benchmarks.md).
Two headlines: splitting a model that fits on one card costs 17–33% throughput,
so dual-GPU is for models that do not fit rather than for speed; and **the CUDA
backend collapses as context grows** — at 32k depth it retains 9% of prompt and
20% of generation throughput, while Vulkan on the same RTX PRO 6000 retains
41% / 90%. (With the CUDA 12.8 container build below, CUDA regains the lead at
every depth and this advice is obsolete.)

The CUDA backend also aborts outright with `-fa off` (`CUDA error: invalid
argument`) at every depth, so it is flash-attention-only here.

That collapse is **proven** to come from the CUDA toolkit used to build
llama.cpp: the identical commit built with CUDA 12.8 (in a container - the host
glibc forbids it natively) has no cliff and is the fastest configuration at
every depth. `./scripts/build-cuda12-container.sh` automates it and produces
`runtime-rocm-cuda128/`, a dual-vendor runtime with the fast CUDA backend.
The patch in [patches/](patches/) is superseded and kept only as a fallback.
Full investigation in [../doc/cuda-fa-blackwell.md](../doc/cuda-fa-blackwell.md).

### Prebuilt binaries

Upstream ships **no Linux CUDA build** - only Windows gets one - so the CUDA
backend must be compiled locally. Ubuntu 26.04 packages llama.cpp but its
`libggml0-backend-*` set is BLAS/HIP/Vulkan only, with no CUDA, because the
toolkit is not redistributable under DFSG.

For the other backends prebuilt binaries do exist and are equivalent to ours:
upstream's `llama-b10331-bin-ubuntu-vulkan-x64` matches this repo's Vulkan build
within 1-2% on both GPUs, and there is a `ubuntu-rocm-7.2-x64` asset too.

**Prebuilt binaries cannot give you dual-vendor.** LM Studio ships an unaffected
CUDA build and a ROCm build at the same commit (`fe2adf0`), but neither exports
`ggml_backend_init`: they are not built with `GGML_BACKEND_DL`, so each bundle is
a closed single-vendor build and the two `.so` files cannot be loaded into one
process. Verified - combining them yields `failed to find ggml_backend_init`.
Upstream's own release assets are likewise one backend per archive.

So a single process driving both an AMD and an NVIDIA GPU has to be built
locally, which is what this repo does, and on Blackwell that means living with
the CUDA toolkit issue above or using Vulkan for the NVIDIA side.

Still unmeasured: a model too large for one card, `--tensor-split` tuning,
depths beyond 32k, quantized KV cache, and concurrent load
(`benchmark-loaded-model.sh` has only been exercised against a mock server).

One unexplained anomaly: in the `rocm-cuda` runtime CUDA0 reports **more free
memory than it has in total** (97249 MiB total, 199823 MiB free). The Vulkan
backend reports the same card sanely (96653 of 97887), so it is specific to the
CUDA backend's reporting. llama.cpp sizes `--fit` from free VRAM, so this needs
explaining before trusting automatic placement on a model that nearly fills the
card.
