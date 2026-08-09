# Multi-GPU LLM

Run llama.cpp across **two GPUs of different vendors at the same time** — AMD (ROCm/HIP or Vulkan) together with NVIDIA (CUDA) — in a single `llama-server` process, without an RPC server.

Most multi-GPU llama.cpp setups assume one vendor. This repo documents what actually happens when you mix them: which backend combinations load, where memory really lands, and which driver bugs you hit on the way.

Both Windows and Linux are covered, on different test machines.

## Pick your platform

| Platform                | Status                        | Scripts    | Start here                             |
|-------------------------|-------------------------------|------------|----------------------------------------|
| **[Windows](windows/)** | Working, benchmarked          | PowerShell | [windows/README.md](windows/README.md) |
| **[Linux](linux/)**     | Scripts ported, not yet built | Bash       | [linux/README.md](linux/README.md)     |

## Test systems

Two physically different machines, with a third planned. This matters for
reading any result in this repo — an APU with unified memory and a pair of
discrete cards behave nothing alike, so **nothing here is currently a clean
Windows-vs-Linux comparison**. Results are keyed by rig, never by OS.

| Rig                        | OS      | CPU / platform               | GPU 1                                    | GPU 2                                 | Memory model                   |
|----------------------------|---------|------------------------------|------------------------------------------|---------------------------------------|--------------------------------|
| **halo-win**               | Windows | AMD Ryzen AI MAX+ 395        | AMD Radeon 8060S iGPU (gfx1151, UMA)     | NVIDIA RTX 5090 (32 GB)               | 128 GB unified, BIOS UMA split |
| **dual-linux**             | Linux   | Intel Core Ultra 7 270K Plus | AMD Radeon AI PRO R9700 (gfx1201, 32 GB) | NVIDIA RTX PRO 6000 Blackwell (96 GB) | Discrete VRAM, no UMA          |
| **halo-linux** *(planned)* | Linux   | AMD Ryzen AI MAX+ 395        | same hardware as halo-win                | same hardware as halo-win             | 128 GB unified, BIOS UMA split |

`halo-linux` will be the Strix Halo box running Linux — the first pairing in
this repo where an OS comparison is actually meaningful, since the hardware is
held constant against `halo-win`.

Full specs, driver versions and how to re-detect them: **[doc/systems.md](doc/systems.md)**.

## Backend combinations

The same four combinations exist on both platforms; only the library extension
and the launcher differ (`.dll` + PowerShell on Windows, `.so` + bash on Linux).

| Backend         | AMD GPU  | NVIDIA GPU | Build requirement           | Notes                                             |
|-----------------|----------|------------|-----------------------------|---------------------------------------------------|
| `rocm-cuda`     | ROCm/HIP | CUDA       | HIP SDK/ROCm + CUDA Toolkit | Usually fastest; most driver-sensitive            |
| `vulkan`        | Vulkan   | —          | Vulkan SDK                  | Single GPU                                        |
| `vulkan-vulkan` | Vulkan   | Vulkan     | Vulkan SDK                  | One backend drives both vendors; avoids ROCm bugs |
| `vulkan-cuda`   | Vulkan   | CUDA       | Vulkan SDK + CUDA Toolkit   | Hybrid compromise                                 |

Each backend is built into its own `runtime-<backend>/` directory, so several
can coexist and you switch between them at launch time rather than rebuilding.

## Shared documentation

These apply across platforms and are the reason both live in one repo:

- **[doc/systems.md](doc/systems.md)** — the test rigs, in detail
- **[doc/benchmarks.md](doc/benchmarks.md)** — results, keyed by rig and backend, plus what the metrics actually mean
- **[doc/rocm-bugs.md](doc/rocm-bugs.md)** — ROCm/HIP memory bugs, with a per-bug matrix of which hardware and OS each one affects
- **[doc/cuda-glibc-243.md](doc/cuda-glibc-243.md)** — why CUDA 13.1 cannot build on Ubuntu 26.04, and what to do instead

## Repository layout

```
.
├── doc/                  # shared, cross-platform
│   ├── systems.md
│   ├── benchmarks.md
│   └── rocm-bugs.md
├── windows/              # PowerShell implementation  (rig: halo-win)
│   ├── scripts/
│   └── patch-system-dll.ps1
└── linux/                # bash implementation        (rig: dual-linux)
    └── scripts/
```

## Status

- [x] Windows: `rocm-cuda`, `vulkan`, `vulkan-vulkan`, `vulkan-cuda` building and serving
- [x] Windows: `isLargeBar` binary patch for >64 GB UMA
- [x] Linux: build and launch scripts ported to bash, detection and error paths verified on the rig
- [x] Linux: confirmed ROCm supports the R9700 natively as `gfx1201` — no `HSA_OVERRIDE_GFX_VERSION` needed
- [x] Linux: confirmed the APU/UMA bugs cannot occur on discrete cards (matrix in `doc/rocm-bugs.md`)
- [x] Linux: `rocm` and `vulkan` backends build and enumerate both GPUs
- [ ] Linux: install CUDA 13.2+ from NVIDIA's repo to unblock `rocm-cuda` / `vulkan-cuda`
- [ ] Linux: tune `--tensor-split` for the 3:1 VRAM asymmetry
- [ ] Benchmarks for each rig
- [ ] Stand up the `halo-linux` rig for a real same-hardware OS comparison
