# Multi-GPU LLM

Run llama.cpp across **two GPUs of different vendors at the same time** — AMD (ROCm/HIP or Vulkan) together with NVIDIA (CUDA) — in a single `llama-server` process, without an RPC server.

Most multi-GPU llama.cpp setups assume one vendor. This repo documents what actually happens when you mix them: which backend combinations load, where memory really lands, and which driver bugs you hit on the way.

Both Windows and Linux are covered, on different test machines.

## Pick your platform

| Platform                | Status               | Scripts    | Start here                             |
|-------------------------|----------------------|------------|----------------------------------------|
| **[Windows](windows/)** | Working, benchmarked | PowerShell | [windows/README.md](windows/README.md) |
| **[Linux](linux/)**     | In progress          | Bash       | [linux/README.md](linux/README.md)     |

## Test systems

Two physically different machines. This matters for reading any result in this
repo — an APU with unified memory and a pair of discrete cards behave nothing
alike, so **nothing here is a clean Windows-vs-Linux comparison**.

| Rig            | OS      | CPU / platform        | GPU 1                                | GPU 2                       | Memory model                   |
|----------------|---------|-----------------------|--------------------------------------|-----------------------------|--------------------------------|
| **halo-win**   | Windows | AMD Ryzen AI MAX+ 395 | AMD Radeon 8060S iGPU (gfx1151, UMA) | NVIDIA RTX 5090 (32 GB)     | 128 GB unified, BIOS UMA split |
| **dual-linux** | Linux   | Intel                 | AMD Radeon AI PRO R9700 (32 GB)      | NVIDIA RTX PRO 6000 (96 GB) | Discrete VRAM, no UMA          |

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
- [ ] Linux: port the build and launch scripts to bash
- [ ] Linux: verify ROCm supports the R9700 (gfx1201) without a `HSA_OVERRIDE_GFX_VERSION` workaround
- [ ] Linux: check whether the Strix Halo memory bugs have any discrete-GPU equivalent
- [ ] Benchmarks for both rigs in a comparable format
