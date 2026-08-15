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
| Total GPU memory              | 96 GB NVIDIA + whatever the UMA split gives the iGPU | 128 GB (32 + 96)              |
| NVIDIA link                   | External OCuLink, PCIe 4.0 x4, ~8 GB/s    | PCIe 5.0 x16, ~64 GB/s                   |
| Largest single-GPU allocation | Limited by GART / `isLargeBar`            | Limited by the card's own VRAM           |
| BIOS UMA tuning               | Required, and consequential               | Not applicable                           |
| Dominant failure mode         | Memory *placement* (spills to shared RAM) | Expected: PCIe topology and split tuning |

Both rigs now run an RTX PRO 6000 96 GB, which makes a cross-rig comparison
tempting and still wrong: one sits in a gen5 x16 slot, the other on four
external lanes with an eighth of the bandwidth. That difference alone reverses
which layout wins, see [benchmarks.md](benchmarks.md).

The practical consequence: the ROCm bugs documented for `halo-win` are APU and
UMA bugs. On `dual-linux` there is no UMA at all, so most of them cannot occur.
See [rocm-bugs.md](rocm-bugs.md) for the per-bug applicability matrix.

## `halo-win` — AMD Strix Halo + RTX PRO 6000 over OCuLink (Windows)

| Component | Details |
|-----------|-----|
| CPU       | AMD Ryzen AI MAX+ 395 (16 cores / 32 threads) |
| RAM       | 128 GB unified (split in BIOS between GPU and OS) |
| GPU 1     | AMD Radeon 8060S iGPU — RDNA 3.5, gfx1151, VRAM via BIOS UMA, 512-bit bus, driver 32.0.31035.1003 |
| GPU 2     | NVIDIA RTX PRO 6000 Blackwell Workstation Edition — 97887 MiB, compute 12.0, driver 595.79, **external over OCuLink, PCIe 4.0 x4** |
| OS        | Windows 11 Pro 26200 |
| AMD stack | HIP SDK 7.1 at `C:\Program Files\AMD\ROCm\7.1` |
| NVIDIA    | driver only — **no CUDA Toolkit installed** (see below) |
| Other     | Vulkan SDK 1.4.341.1, MSVC 14.44 and 14.51, VS 18 Enterprise + VS 2022 BuildTools |

### Hardware history

The discrete card was **swapped on 2026-08-15**: RTX 5090 32 GB → RTX PRO 6000
96 GB. Anything in this repo recorded against `halo-win` before that date was
measured with the 5090; the AMD-side observations (UMA bugs, the `isLargeBar`
patch) are properties of the APU and carry over, the NVIDIA-side ones do not.

The swap makes this rig share its discrete GPU model with `dual-linux`, which is
the first time two rigs here hold anything constant — but they hold it on very
different links, see below.

The card was **on a Thunderbolt 5 tunnel until later the same day**, when
OCuLink was made to work. Results tagged TB5 in [benchmarks.md](benchmarks.md)
are from that window.

### Verified on this rig

- **The NVIDIA card is external, and only four lanes wide.** OCuLink, trained at
  `gen4 x4` under load — ~8 GB/s against the PCIe 5.0 x16 (~64 GB/s) the
  identical card gets on `dual-linux`. **This is the single most consequential
  fact about the rig**: any layout that streams weights across it during prefill
  is starved, see [benchmarks.md](benchmarks.md).
- **`pcie.link.gen.current` reads 1 at idle.** The link downclocks when nothing
  is running, so a bare `nvidia-smi` on an idle box reports gen1 and looks
  alarming. Sample it *during* a prefill — 14 of 14 samples read `gen4, width 4`.
  Over the earlier TB5 tunnel it read gen4 even at idle, which is the opposite
  of what one would guess.
- **OCuLink only enumerates with Resizable BAR disabled in BIOS.** With ReBAR on
  it did not come up at all, which is why this rig ran on Thunderbolt first. The
  practical loss is nil here: HIP reported `isLargeBar: 0` on the iGPU either
  way, and llama.cpp does not need a large BAR on the NVIDIA card.
- **No CUDA Toolkit is installed.** `C:\Program Files\NVIDIA GPU Computing
  Toolkit\CUDA\v13.2` and `v13.3` exist but contain only empty `bin`/`lib`
  shells — leftovers of a runtime, with no `nvcc`. `setup-llama.ps1` therefore
  fails at `Require-Command nvcc` for any backend containing CUDA, and the CUDA
  backend has to come from an upstream prebuilt release instead.
- **The upstream Windows ROCm build does not work here, and not for the obvious
  reason.** `llama-b10441-bin-win-rocm-7.14-x64.zip` lists no devices at all,
  yet its `ggml-hip.dll` *does* contain gfx1151 (the baked target list runs
  gfx1010 → gfx1250). `hipInfo.exe` from the SDK enumerates the iGPU fine, so
  HSA works; forcing the SDK's `amdhip64_7.dll` next to the exe changes nothing.
  It is an ABI mismatch between that build's ROCm 7.14 and the installed HIP SDK
  7.1. A local single-arch build of just `ggml-hip.dll` (69 MB against the
  prebuilt's 881 MB) enumerates `ROCm0` immediately.
- **ROCm's clang cannot link on its own.** Building the HIP backend outside a
  Visual Studio developer environment dies with `lld-link: could not open
  'msvcrtd.lib' / 'oldnames.lib'`. Import `vcvars64.bat -vcvars_ver=14.44`
  first — the toolset that `setup-llama.ps1` already selects for CUDA.
- **A HIP runtime that lists no devices is usually a `PATH` problem, not an ABI
  one.** `ggml-hip.dll` resolves rocBLAS and its Tensile kernel library out of
  the HIP SDK, which is not on the system `PATH`. Without it the runtime loads
  and reports `CUDA0` only — the same silent symptom as the ABI mismatch above.
  `start-llama-server.ps1` now prepends `$HIP_PATH\bin` for ROCm modes.
- **Pinned host memory has to be set per layout, and each setting breaks the
  other.** `GGML_CUDA_NO_PINNED=1` (the launcher's default for CUDA/HIP modes)
  makes the dual layouts fail or hang on the iGPU's single large allocation;
  leaving it unset makes the CUDA-only layout fail on a ~58 GiB *pinned* host
  buffer, regardless of how much RAM is free. `start-llama-server.ps1` takes
  `-AllowPinned`; the DeepSeek profile decides it per layout. Details in
  [benchmarks.md](benchmarks.md).
- **Measured RAM bandwidth: 83.8 GB/s** at 16 threads with
  `windows/scripts/membw.c`, the port of the Linux benchmark (same buffer size
  and repeat count, so the rigs compare directly) — against **89.9 GB/s** on
  `dual-linux`. The LPDDR5X is *not* faster than that rig's DDR5 by this
  measure, which cost an earlier revision of [benchmarks.md](benchmarks.md) an
  explanation. The benchmark is thread-limited rather than bus-limited: 54.3
  GB/s on 8 threads, 83.8 on 16, 95.4 on 24, 102.5 on 32 — still rising when it
  runs out of logical CPUs, so treat it as a comparable lower bound rather than
  the machine's ceiling. Run it with `windows/scripts/run-membw.ps1`.
- **Port 8080 is occupied** by a service called `AgentService`. Everything here
  runs on 8090.

### BIOS framebuffer: leave it at 1 GB

Settled by measurement, and the answer is the counter-intuitive one — **the
smallest carve-out is best for every layout tested here**:

| Framebuffer | Windows sees | `hipInfo` total | ROCm0 / Vulkan0 total | `vulkan-cuda` | `rocm-cuda` |
|---|---|---|---|---|---|
| 1 GB   | 126.6 GB RAM | 76.99 GB | 78836 / 104773 MiB | **352 pp / 34.4 tg** | **497 pp / 36.1 tg** |
| 64 GB  | 63.6 GB RAM  | 99.74 GB | 102129 / 114326 MiB | 188 pp / 31.1 tg | 468 pp / 32.6 tg |

At 64 GB the iGPU's memory is a dedicated carve-out that is not fully
CPU-visible (`isLargeBar: 0`), so host writes into it go through a small BAR
window and a staging buffer. The Vulkan backend pays 45% of its prefill for
that; HIP does not pay it at all, which is why the carve-out looked mandatory
for HIP until it was tested without one. At 1 GB the iGPU runs out of ordinary
system RAM through GTT, which is fully CPU-visible and needs no staging — and
HIP allocates its 57.4 GiB there without complaint.

A large carve-out costs twice over: the OS is left with 63.6 GB, which is no
longer enough for the CUDA-only layout's ~56 GB of RAM-hosted experts.

The older guidance — 64 GB UMA conservative, 96 GB UMA broken — was written for
ROCm running a model *entirely* on the iGPU, and still applies to that case.
It does not describe a dual layout where the iGPU is fed from another device.
Details in [rocm-bugs.md](rocm-bugs.md).

## `dual-linux` — Radeon AI PRO R9700 + RTX PRO 6000 (Linux)

| Component | Details |
|-----------|-----|
| CPU       | Intel Core Ultra 7 270K Plus (Arrow Lake-S) |
| RAM       | 122 GiB usable — 2×64 GB (Corsair 6400C42) at **7400 MT/s** (previously 2×48+2×64 mixed at 6267) |
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
- **Measured RAM bandwidth: 89.9 GB/s** after the 4→2 DIMM swap (2×64 GB at
  7400 MT/s — 1000 above the kit's 6400 C42 rating, the freed IMC took the
  overclock). Before: 74.6 GB/s with mixed kits at 6267 MT/s; +20.5%. Measured
  with `linux/scripts/membw.c` (16-thread streaming read). This is the number
  that prices CPU expert offload — each RAM-hosted MXFP4 expert layer costs
  ~0.85 ms/token against it. The capacity cost: 215→122 GiB means the 146 GB
  MXFP4 model no longer fits the page cache, so the first prefill after a
  model load pays NVMe page-in for the CPU-hosted experts (visible as a
  halved pp on the first request; steady state is unaffected).
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
