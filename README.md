# Multi-GPU LLM

Run llama.cpp across **two GPUs of different vendors at the same time** — AMD (ROCm/HIP or Vulkan) together with NVIDIA (CUDA) — in a single `llama-server` process, without an RPC server.

Most multi-GPU llama.cpp setups assume one vendor. This repo documents what actually happens when you mix them: which backend combinations load, where memory really lands, and which driver bugs you hit on the way.

Both Windows and Linux are covered, on different test machines.

> ### Read this if you have a Blackwell NVIDIA GPU
>
> On stock llama.cpp, the **CUDA backend loses 50-80% of its token-generation
> speed the moment the context passes 8192 tokens**. Measured on an RTX PRO 6000
> (sm_120): 72 -> 35 t/s in one step for a 27B model, 32 -> 9 t/s for a 70B one.
> The GPU sits at ~210 W instead of ~410 W. It is not a hardware fault and not
> your driver.
>
> The cause is **proven to be the CUDA toolkit used to build llama.cpp** - not
> llama.cpp itself and not your hardware. The identical commit built with
> CUDA 12.8 shows no collapse (74.6 t/s at 8k where the 13.3 build drops to
> 34.7) and is the fastest option at every depth. There is no official Linux
> CUDA binary from upstream, so anyone on Linux builds their own and can hit
> this. In order of preference:
>
> - build the CUDA backend with **CUDA 12.8 in a container**:
>   `./scripts/build-cuda12-container.sh` does it and assembles a dual-vendor
>   runtime (`runtime-rocm-cuda128/`) around it, or
> - drive the NVIDIA card with **Vulkan** instead of CUDA (unaffected, ~3% cost
>   at short context), or
> - as a last resort `--patches` ([linux/patches/](linux/patches/)) recovers
>   3.3-4.2x generation on a CUDA 13.3 build, but not prefill.
>
> Full analysis, measurements and reproduction: **[doc/cuda-fa-blackwell.md](doc/cuda-fa-blackwell.md)**.

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
- **[doc/cuda-glibc-243.md](doc/cuda-glibc-243.md)** — why the distro CUDA 13.1 cannot build on Ubuntu 26.04, and how to fix it
- **[doc/cuda-fa-blackwell.md](doc/cuda-fa-blackwell.md)** — CUDA token generation collapses at 8192 context on Blackwell: cause, measurements, and a one-line fix

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
- [x] Linux: all four backends build; `rocm-cuda` runs ROCm and CUDA in one process
- [x] Linux: benchmark matrix across all seven single/dual configurations
- [x] Linux: context-depth sweep — found the CUDA backend collapses past ~8k while Vulkan does not
- [x] Linux: root-caused that collapse to an Ada-tuned FA heuristic applied to Blackwell; `--patches` restores 2-5x
- [ ] Linux: benchmark a model too large for one card — the case dual-GPU exists for
- [ ] Linux: tune `--tensor-split` (everything so far uses the proportional default)
- [ ] Linux: tune `--tensor-split` for the 3:1 VRAM asymmetry
- [ ] Benchmarks for each rig
- [ ] Stand up the `halo-linux` rig for a real same-hardware OS comparison
