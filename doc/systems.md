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

| Component | Details                                                  |
|-----------|----------------------------------------------------------|
| CPU       | Intel — *fill in exact model*                            |
| RAM       | *fill in*                                                |
| GPU 1     | AMD Radeon AI PRO R9700 32 GB (Gigabyte AI TOP) — RDNA 4 |
| GPU 2     | NVIDIA RTX PRO 6000 96 GB — Blackwell                    |
| OS        | Linux — *fill in distro + kernel*                        |
| AMD stack | ROCm — *fill in version*                                 |
| NVIDIA    | Driver + CUDA Toolkit — *fill in versions*               |

Both cards are discrete, so there is no UMA setting and no BIOS memory split.
128 GB of combined dedicated VRAM is more than `halo-win` has in total, and all
of it is real VRAM rather than carved-out system RAM.

Open questions for this rig, to be answered by testing:

- **gfx target of the R9700.** Expected `gfx1201` (RDNA 4). Confirm with
  `rocminfo`, and check whether the installed ROCm version lists it as
  officially supported — if not, `HSA_OVERRIDE_GFX_VERSION` may be needed, which
  changes the build and launch procedure.
- **PCIe topology.** How many lanes each card gets, and whether they sit behind
  the same root complex. Tensor-split tuning depends on this far more here than
  on `halo-win`, where one "GPU" was on-die.
- **Driver coexistence.** `amdgpu` and the proprietary NVIDIA driver loaded
  simultaneously, with both ROCm and CUDA runtimes initialised in one process.

### Filling in the unknowns

```bash
# CPU / RAM / kernel / distro
lscpu | grep -E 'Model name|^CPU\(s\)'
free -h
uname -r
. /etc/os-release && echo "$PRETTY_NAME"

# AMD: gfx target, VRAM, ROCm version
rocminfo | grep -E 'Name:|gfx|Marketing'
rocm-smi --showproductname --showmeminfo vram
cat /opt/rocm/.info/version

# NVIDIA: model, VRAM, driver, CUDA
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
nvcc --version | tail -2

# PCIe topology and negotiated link width
lspci -nn | grep -Ei 'vga|3d|display'
nvidia-smi --query-gpu=name,pcie.link.width.current,pcie.link.gen.current --format=csv
```

Paste the results into the table above, replacing the *fill in* placeholders.
