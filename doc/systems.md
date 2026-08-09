# Test systems

Everything in this repo was produced on one of two machines. They differ in
almost every dimension that matters for multi-GPU LLM inference, so always check
which rig a result came from before generalising it.

Rigs are referred to by short name — **`halo-win`** and **`dual-linux`** — in
benchmarks and bug reports.

## Why the distinction matters

| Property                      | `halo-win`                                | `dual-linux`                             |
|-------------------------------|-------------------------------------------|------------------------------------------|
| AMD GPU type                  | Integrated (APU)                          | Discrete                                 |
| AMD memory                    | Shared with system RAM via BIOS UMA split | Dedicated VRAM                           |
| Total GPU memory              | ~96 GB (64 UMA + 32)                      | 128 GB (32 + 96)                         |
| Largest single-GPU allocation | Limited by GART / `isLargeBar`            | Limited by the card's own VRAM           |
| BIOS UMA tuning               | Required, and consequential               | Not applicable                           |
| Dominant failure mode         | Memory *placement* (spills to shared RAM) | Expected: PCIe topology and split tuning |

The practical consequence: the ROCm bugs documented for `halo-win` are APU and
UMA bugs. On `dual-linux` there is no UMA at all, so most of them cannot occur.
See [rocm-bugs.md](rocm-bugs.md) for the per-bug applicability matrix.

## `halo-win` — AMD Strix Halo + RTX 5090 (Windows)

| Component | Details |
|-----------|-----|
| CPU       | AMD Ryzen AI MAX+ 395 (32 threads) |
| RAM       | 128 GB unified (split in BIOS between GPU and OS) |
| GPU 1     | AMD Radeon 8060S iGPU — RDNA 3.5, gfx1151, VRAM via BIOS UMA, 512-bit bus |
| GPU 2     | NVIDIA GeForce RTX 5090 — 32 GB, compute 12.0, PCIe 4.0 x4 |
| OS        | Windows 11 |
| AMD stack | HIP SDK 7.1.1 (AMD Software PRO Edition 26.Q1) |
| NVIDIA    | CUDA Toolkit 13.2 |

BIOS UMA split is the single most important setting on this rig. 64 GB UMA
(64 GPU / 64 OS) is the conservative baseline; 96 GB UMA triggers the memory
bugs. Details in [rocm-bugs.md](rocm-bugs.md).

## `dual-linux` — Radeon AI PRO R9700 + RTX PRO 6000 (Linux)

| Component | Details |
|-----------|-----|
| CPU       | Intel Core Ultra 7 270K Plus (Arrow Lake-S) |
| RAM       | 215 GiB usable — 2×48 GB (Corsair 6000C30) + 2×64 GB (Corsair 6400C42), all at 6267 MT/s |
| GPU 1     | AMD Radeon AI PRO R9700 (Gigabyte AI TOP) — RDNA 4, Navi 48, gfx1201, 31.9 GiB |
| GPU 2     | NVIDIA RTX PRO 6000 Blackwell Workstation Edition — 95.6 GiB, compute 12.0, PCIe 5.0 x16 |
| OS        | Ubuntu 26.04 LTS, kernel 7.0.0-29-generic |
| AMD stack | ROCm 7.1 (distro packages — `libamdhip64-7` 7.1.0, `rocminfo`/`rocm-smi` 7.1.1) |
| NVIDIA    | Driver 595.84, CUDA Toolkit 13.3 at `/usr/local/cuda-13.3` (NVIDIA repo) |
| Vulkan    | RADV (AMD), NVIDIA ICD, plus Intel ARL iGPU and llvmpipe |

Both cards are discrete, so there is no UMA setting and no BIOS memory split.
128 GB of combined dedicated VRAM is more than `halo-win` has in total, and all
of it is real VRAM rather than carved-out system RAM.

### Verified on this rig

- **gfx1201 is natively supported.** `rocminfo` reports `gfx1201` and
  `amdgcn-amd-amdhsa--gfx1201` directly, so **no `HSA_OVERRIDE_GFX_VERSION`
  workaround is needed**. `setup-llama.sh` reads the target from `rocminfo` and
  passes it as `-DAMDGPU_TARGETS`.
- **ROCm lives in `/usr`, not `/opt/rocm`.** Ubuntu 26.04 ships ROCm as distro
  packages; there is no `/opt/rocm` directory at all, and the CMake config sits
  under `/usr/lib/x86_64-linux-gnu/cmake/hip`. Scripts must handle both layouts.
- **The CUDA toolkit in `PATH` is too old for this GPU.** `/usr/bin/nvcc` is
  CUDA 12.4, which tops out at `compute_90` — the RTX PRO 6000 is `sm_120`.
  CUDA 13.1 is installed alongside at `/usr/local/cuda-13.1` and does support
  `compute_120`, but it is not first in `PATH`. Building without pinning the
  right toolkit fails with "unsupported gpu architecture". `setup-llama.sh`
  detects the GPU's compute capability and picks a toolkit that supports it —
  the Linux analogue of the MSVC toolset workaround on Windows.
- **Four Vulkan devices are visible, not two.** RADV (AMD) and the NVIDIA ICD,
  but also the Intel Arrow Lake iGPU and the llvmpipe software rasteriser.
  Selecting Vulkan devices by index would pick the wrong pair, so
  `start-llama-server.sh` matches on the device description instead.

### Still open

- **PCIe topology for the AMD card.** The RTX PRO 6000 negotiates gen5 x16.
  The R9700's link width has not been read yet (needs root for `lspci -vv`).
  Tensor-split tuning depends on this far more here than on `halo-win`, where
  one "GPU" was on-die.
- **`--tensor-split` ratio.** 96 GB vs 32 GB is roughly 3:1, so an even split
  wastes most of the NVIDIA card. Starting point is `1,3` (AMD first — that is
  the order the launcher builds `--device` in), to be tuned per model.
- **The distro CUDA is unusable; NVIDIA's is fine.** Ubuntu's CUDA 13.1 cannot
  compile against glibc 2.43 — see [cuda-glibc-243.md](cuda-glibc-243.md).
  CUDA 13.3 from NVIDIA's apt repo works, and all four backends build with it.
- **Driver coexistence under load.** ROCm and CUDA now initialise together in a
  single process and both cards enumerate, but no model has been loaded, so the
  combination is untested under real memory pressure.
- **Measured RAM bandwidth: 74.6 GB/s** (16-thread streaming read, ~75% of
  the 6267 MT/s dual-channel theoretical). This is the number that prices CPU
  expert offload in the DeepSeek profiles — each RAM-hosted expert layer costs
  ~1 ms/token against it. Re-measure after any DIMM change
  (`scratchpad membw.c`; a planned 4→2 DIMM swap to the 64 GB pair trades
  ~96 GB of capacity for whatever clock the freed IMC allows — the kept kit is
  rated 6400 C42, so gains beyond ~2% require overclocking past rating).
- **CUDA0 misreports free memory.** In the `rocm-cuda` runtime it claims
  199755 MiB free out of 97249 MiB total. Since llama.cpp sizes `--fit` from
  free VRAM, this needs explaining before trusting automatic placement.

### Re-detecting all of this

```bash
# CPU / RAM / kernel / distro
lscpu | grep -E 'Model name|^CPU\(s\)'
free -h
uname -r
. /etc/os-release && echo "$PRETTY_NAME"

# AMD: gfx target, VRAM, ROCm version
rocminfo | grep -E 'Name:|gfx|Marketing'
rocm-smi --showproductname --showmeminfo vram
cat /opt/rocm/.info/version 2>/dev/null || dpkg -l | grep -E 'rocminfo|libamdhip'

# NVIDIA: model, VRAM, driver, CUDA
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
nvcc --version | tail -2                  # NOTE: may not be the toolkit that gets used
ls -d /usr/local/cuda*/bin/nvcc            # other toolkits installed alongside

# PCIe topology and negotiated link width
lspci -nn | grep -Ei 'vga|3d|display'
nvidia-smi --query-gpu=name,pcie.link.width.current,pcie.link.gen.current --format=csv
```

`./linux/scripts/list-devices.sh` prints most of this in one go, and works
before anything has been built.

## Planned: `halo-linux` — Strix Halo on Linux

The Strix Halo machine will eventually also run Linux, giving a third rig and
the first genuine same-hardware OS comparison in this repo.

That combination re-activates documentation that is dormant for `dual-linux`:
the APU and UMA bugs in [rocm-bugs.md](rocm-bugs.md) apply again, and the TTM
kernel-parameter workarounds described there — irrelevant to a discrete card —
become the primary tuning lever. Benchmarks against `halo-win` will be directly
comparable, which no pair of current rigs is.
