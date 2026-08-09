# Benchmarks

Results are grouped by **rig**, because the two test machines are not
comparable hardware — see [systems.md](systems.md). A number from `halo-win`
says nothing about what `dual-linux` will do, and vice versa. Treat any
cross-rig table as "two different machines", never as "Windows vs Linux".

Always record alongside a result: rig, backend mode, model and quant, context
size, `--tensor-split`, and any relevant BIOS/driver setting.

## Understanding the metrics

When looking at results (especially inside `benchmark.log`), you will see the following key metrics:

* **`PromptTokens` (Prefill Tokens):** The number of tokens in the initial input prompt sent to the model. The model processing these tokens all at once is known as the "prefill" phase.
* **`PredictTokens` (Generated Tokens):** The number of new tokens the model generated in response to the prompt. Generating these one by one is known as the "decoding" phase.
* **`GenTokPerSec` (Generation Speed):** This is the raw generation speed reported by the backend server. It calculates how fast tokens are generated *only during the decoding phase*. It completely excludes the time spent processing the initial prompt (prefill) and any network latency.
* **`TotalTokPerSec` (Total Throughput):** This represents the *end-to-end* throughput from the client's perspective (`Total Generated Tokens / Total Wall Time`). Because the wall time includes the HTTP overhead, the potentially long prompt processing (prefill) phase, and the actual generation phase, this number will always be lower than `GenTokPerSec`. Even when running with `--parallel 1` (a single request), `TotalTokPerSec` will be lower because the prompt prefill time drags down the overall average speed of the request lifecycle.

## Benchmarking tools

Two tools, measuring different things. Each exists per platform with the same
name and behaviour — `.ps1` under [../windows/scripts/](../windows/scripts/),
`.sh` under [../linux/scripts/](../linux/scripts/).

### 1. `run-llama-bench` — precise hardware testing

Uses the internal `llama-bench` tool. Best for measuring absolute maximum hardware capability, ideal parallel layer splits, and raw memory bandwidth. It executes the model locally in C++, completely bypassing HTTP, network latency, and the server queuing system. You can test parallel processing perfectly using batch sizes (`-b 1,2,4`).

* **Usage:** run the script directly. If `llama-server` is already running, it will automatically detect the parameters used (model path, layers, tensor splits, etc.).
* **Important:** If your model fills your VRAM, you should close `llama-server` before actually running the parsed command to avoid Out-Of-Memory (OOM) errors, as both programs will fight for the same GPU memory.

### 2. `benchmark-loaded-model` — real-world HTTP testing

Tests the true end-to-end experience of a client interacting with `llama-server` over the REST API. It accounts for web-server threading, connection handling, HTTP overhead, and the continuous KV-cache context switching that happens when multiple real clients hit the server simultaneously (`--parallel X`).

* **Usage:** Start your model server, then run the script. It will automatically detect the running `llama-server` to extract the mode and extra args, and log the results into `benchmark.log`.

---

## Results: `halo-win` (Strix Halo iGPU + RTX 5090, Windows)

Pending. To be measured across:

1. `rocm-cuda`
2. `vulkan-vulkan`
3. `vulkan-cuda`

| Model | Quant | Backend | Context | Tensor split | UMA | GenTok/s | TotalTok/s |
|-------|-------|---------|---------|--------------|-----|---------:|-----------:|
| —     | —     | —       | —       | —            | —   | —        | —          |

## Results: `dual-linux` (Radeon AI PRO R9700 + RTX PRO 6000, Linux)

**Model:** Qwen3.6-27B-uncensored-heretic-v2 i1-Q6_K (21 GB) — chosen because it
fits entirely on either card alone, so single-GPU and dual-GPU are directly
comparable.
**Tool:** `llama-bench`, `-p 512 -n 128 -r 3`, defaults otherwise (flash-attn
auto, f16 KV, layer split, proportional tensor split).
**Build:** llama.cpp `7ba604f`, CUDA 13.3, ROCm 7.1, RADV, **stock (no
`--patches`)**. Measured 2026-08-09.

| Config          | Backend / devices      | pp512 (t/s)      | vs CUDA | tg128 (t/s)     | vs CUDA |
|-----------------|------------------------|-----------------:|--------:|----------------:|--------:|
| **cuda**        | CUDA0 (NVIDIA alone)   | **2232.6** ±28.5 | 100%    | **61.17** ±0.06 | 100%    |
| vulkan (NVIDIA) | Vulkan1 (NVIDIA alone) | 2168.5 ±11.0     | 97%     | 59.47 ±0.20     | 97%     |
| rocm-cuda       | ROCm0 + CUDA0          | 1635.6 ±35.9     | 73%     | 50.58 ±0.05     | 83%     |
| vulkan-vulkan   | Vulkan2 + Vulkan1      | 1514.2 ±6.8      | 68%     | 42.66 ±0.14     | 70%     |
| vulkan-cuda     | Vulkan2 + CUDA0        | 1506.3 ±41.8     | 67%     | 49.56 ±0.07     | 81%     |
| vulkan (AMD)    | Vulkan2 (AMD alone)    | 869.9 ±1.2       | 39%     | 25.37 ±0.01     | 41%     |
| rocm (AMD)      | ROCm0 (AMD alone)      | 743.5 ±7.0       | 33%     | 24.18 ±0.02     | 40%     |

### What this says

**Splitting a model that fits on one card makes it slower, not faster.** The
best dual configuration (`rocm-cuda`) reaches 73% of prompt throughput and 83%
of generation throughput of the RTX PRO 6000 on its own. This is inherent to
layer splitting: only one GPU computes at a time while the other waits, and
activations cross PCIe at every boundary. Adding the slower card drags the
average toward it.

The practical rule for this rig: **use dual-GPU only when the model does not fit
in 96 GB.** Below that, single-GPU CUDA wins on every axis. Above it, dual is
the only option and the comparison stops mattering.

**Vulkan on NVIDIA costs almost nothing** — 97% of CUDA on both metrics. If a
CUDA toolchain is unavailable (see [cuda-glibc-243.md](cuda-glibc-243.md)),
`vulkan` is a near-free substitute on this card.

**Vulkan beats ROCm on the R9700.** RADV is 17% faster at prompt processing
(869.9 vs 743.5) and 5% faster at generation. On gfx1201 the ROCm backend is not
the fast path, which inverts the assumption carried over from `halo-win`, where
ROCm is the performance option and Vulkan the safe fallback.

**Among dual configurations the backend matters more for generation than for
prefill.** All three land within 8% on pp512, but tg128 spans 42.7–50.6 —
`vulkan-vulkan` is the weakest, and pairing either AMD backend with CUDA on the
NVIDIA side recovers most of the gap.

### Context-depth sweep

> These numbers are **stock llama.cpp, without `--patches`**. The CUDA collapse
> shown here is a llama.cpp bug that this repo ships a fix for; a build made
> with `--patches` does not have it. See
> [cuda-fa-blackwell.md](cuda-fa-blackwell.md).

Same model and rig across KV-cache depth, `-fa on` for every backend so they all
run the same attention algorithm, `-r 2`. Depth 0 is re-measured here and is
therefore not directly comparable with the `-fa auto` table above.

**Token generation (tg128, t/s)**

| Config          | d=0  | d=4096 | d=16384 | d=32768  | retained |
|-----------------|-----:|-------:|--------:|---------:|---------:|
| vulkan (NVIDIA) | 61.2 | 60.2   | 58.0    | **55.3** | 90%      |
| vulkan-vulkan   | 42.8 | 42.1   | 40.3    | 37.2     | 87%      |
| vulkan (AMD)    | 25.4 | 25.0   | 24.1    | 23.1     | 91%      |
| rocm (AMD)      | 24.2 | 24.0   | 23.3    | 22.3     | 92%      |
| rocm-cuda       | 50.6 | 49.2   | 21.1    | 13.4     | 26%      |
| cuda (NVIDIA)   | 64.7 | 62.4   | 21.3    | 12.9     | **20%**  |
| vulkan-cuda     | 49.6 | 47.4   | 19.4    | 12.1     | 24%      |

Prompt processing degrades in the same shape: at 32k, `vulkan (NVIDIA)` holds
941.5 t/s (41% of its depth-0 rate) against `cuda` at 205.7 (9%).

**Everything containing CUDA collapses; nothing else does.** That is the whole
story of this table, and it is a llama.cpp bug rather than a property of the
hardware -- [cuda-fa-blackwell.md](cuda-fa-blackwell.md) has the cause, the
exact 8192 threshold, the power measurements and the fix. Building with
`--patches` recovers 3.3-4.2x at 32k on every CUDA configuration, without
changing the non-CUDA ones.

Two things in this table are *not* about that bug and are worth keeping:

- **The ranking inverts between short and long context.** At depth 0 `rocm-cuda`
  is the best dual configuration; by 32k `vulkan-vulkan` is roughly 3x better.
  Backend choices made on depth-0 numbers are wrong for long-context work.
  (This inversion disappears with a healthy CUDA build - see below.)
- **Vulkan and ROCm both degrade gracefully**, holding 87-92% of generation
  throughput out to 32k on either vendor.

### The definitive matrix: CUDA backend built with CUDA 12.8

Same model, depths and parameters, with the CUDA-containing configurations
running the `build-cuda12-container.sh` runtimes (`runtime-rocm-cuda128`,
`runtime-vulkan-cuda128`). Vulkan and ROCm rows use the same binaries as
before and moved less than 1%, which validates the comparison.

**Token generation (tg128, t/s)** - arrow shows CUDA-13.3-stock -> CUDA-12.8:

| Config               | d=0      | d=4096   | d=32768             |
|----------------------|---------:|---------:|--------------------:|
| cuda (NVIDIA)        | 78 -> 77 | 75 -> 76 | 13 -> **70** (5.2x) |
| vulkan (NVIDIA)      | 73       | 72       | 65                  |
| rocm-cuda (dual)     | 59 -> 51 | 57 -> 51 | 14 -> **47** (3.4x) |
| vulkan-cuda (dual)   | 58 -> 53 | 56 -> 52 | 13 -> **45** (3.5x) |
| vulkan-vulkan (dual) | 52       | 51       | 44                  |
| vulkan (AMD)         | 31       | 30       | 27                  |
| rocm (AMD)           | 28       | 28       | 26                  |

**Prompt processing (pp512, t/s)**:

| Config               | d=0                     | d=4096                  | d=32768                 |
|----------------------|------------------------:|------------------------:|------------------------:|
| cuda (NVIDIA)        | 2787 -> **3597** (1.3x) | 1106 -> **3863** (3.5x) | 209 -> **2936** (14.1x) |
| vulkan (NVIDIA)      | 2663                    | 2166                    | 989                     |
| rocm-cuda (dual)     | 2063 -> 2153            | 1019 -> **2012** (2.0x) | 222 -> **1277** (5.8x)  |
| vulkan-cuda (dual)   | 1703 -> 1738            | 833 -> **1407** (1.7x)  | 180 -> **575** (3.2x)   |
| vulkan-vulkan (dual) | 1722                    | 1423                    | 665                     |
| vulkan (AMD)         | 993                     | 885                     | 529                     |
| rocm (AMD)           | 1152                    | 988                     | 528                     |

What changes with the healthy build:

- **CUDA is simply the fastest NVIDIA backend at every depth**, in both
  metrics. The short/long-context ranking inversion is gone; so is the earlier
  advice to prefer Vulkan for long context. Vulkan remains the fallback when a
  container build is not an option.
- **The 13.3 damage was never limited to generation past 8k.** Prefill was
  losing 1.3x at depth 0, 3.5x at 4k and 14x at 32k - a penalty invisible in
  the earlier tables because every configuration shared it.
- **`rocm-cuda` is again the best dual configuration at every depth**
  (51/51/47 vs 52/50/44 for `vulkan-vulkan`, and roughly double its prefill).
- **One real trade-off**: the mixed dual runtimes lose ~10% generation at short
  context versus the pure 13.3 build (59 -> 51 at d=0, reproduced across both
  dual configs). Unexplained; if a workload never exceeds ~4k context, the
  stock 13.3 dual is marginally faster.

### Bug: the CUDA backend cannot run with flash attention disabled

`-fa off` aborts immediately on the CUDA backend, at every depth including 0:

```
CUDA error: invalid argument
  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2374
```

Reproduced on both test models. In practice the CUDA backend on this card is
flash-attention-only, which also means the depth-0 numbers in the first table
(`-fa auto`) must have had FA enabled — otherwise they would have crashed too.

Investigated — see [cuda-fa-blackwell.md](cuda-fa-blackwell.md). It is llama.cpp
code, not the CUDA toolkit and not a hardware fault.

### Not yet measured

- A model too large for one card — the case dual-GPU actually exists for.
- `--tensor-split` tuning; everything above uses the proportional default.
- Depths beyond 32k, quantized KV cache, and concurrent requests
  (`benchmark-loaded-model.sh` has not been run against a real model).

## Results: `dual-linux` — DeepSeek V4 Flash (the model dual-GPU exists for)

DeepSeek V4 Flash 0731 UD-IQ3_XXS: 98 GB of weights, 43 layers, 256 experts,
MLA attention. MLA compresses the KV cache to ~50 KB/token across all layers,
so even 256k context is only ~13 GB — the whole thing fits in 96 + 32 GB with
room to spare, and the KV is preallocated so a filling context cannot OOM.

Measured over HTTP (single request, 128 predicted tokens, CUDA 12.8 runtimes):

| Config | metric | 4k | 16k | 61–65k | 130k |
|---|---|---:|---:|---:|---:|
| rocm-cuda dual | pp t/s | 1106 | 909 | 636 | 417 |
| cuda + n-cpu-moe 8 | pp t/s | 936 | 986 | **876** | **766** |
| rocm-cuda dual | tg t/s | **45.3** | **42.9** | **38.2** | **32.9** |
| cuda + n-cpu-moe 8 | tg t/s | 29.3 | 29.7 | 27.7 | 25.9 |

Two different winners: the dual generates ~1.4x faster (weights stream from two
memory buses), but CUDA-only wins prefill beyond ~16k — in dual layer-split the
fast card waits for the slow one on every step, and prefill is where that hurts.

### Full 256k, verified on two paths

Both verified with two consecutive 261,900-token prompts including a
full-cache clear between them, no faults:

| Path | Config | pp t/s | tg @ 256k | 256k prefill |
|---|---|---:|---:|---:|
| IQ3_XXS, CUDA-only | `-c 262144 --n-cpu-moe 10` | 531 / 522 | 22.0 | ~8.3 min |
| MXFP4, expert-offload dual | `-c 262144 --n-cpu-moe 12` + `-ot` | 386 / 385 | 18.4 / 18.3 | ~11.4 min |

The MXFP4 run additionally survived a client interrupted mid-prefill (the
server then cleared the partial state and refilled without incident).

### Stability: every dual layout is currently broken for this model

The ROCm backend (gfx1201) faults intermittently under this model's compute:

- `HSA_STATUS_ERROR_MEMORY_FAULT` mid-prefill at ~43k tokens (fresh cache,
  `-c 98304`) — yet a full 130k prefill passed at `-c 131072`. Not a size
  threshold; probabilistic with the amount of work done.
- Crash in the first decode after clearing a full ~130k cache.
- The decisive experiment: an `-ot` layout with **only expert FFN weights** on
  the AMD card (`-ts 0,1 -ot 'blk\.(3[1-9]|4[0-2])\.ffn_.*_exps.*=ROCm0'`) —
  no KV, no attention, no DSV4 cache structures on ROCm — still faulted, at
  225k tokens into a 262k prefill. That acquits the DSV4 cache code (blamed
  here in an earlier revision) and points at the **HIP MoE expert-matmul /
  IQ3_XXS i-quant path**. The same card ran hours of dense Q4/Q6 Qwen and
  Hermes benchmarks on ROCm without a single fault.
- The expert-offload layout is otherwise excellent while it lasts: 57.0 t/s
  generation at short context — faster than the classic dual (47.4), because
  the NVIDIA card runs attention while the AMD card serves expert matmuls from
  VRAM in parallel instead of the cards taking turns. Worth revisiting once
  the HIP fault is fixed upstream.
- Vulkan is no refuge either: `vk::ErrorDeviceLost` ~20k tokens into prefill
  with a 256k allocation, and ~19 t/s MoE generation on RADV.

CUDA-only passed every killer scenario (130k and 256k double-probes with
full-cache clears), and the expert-offload dual passed them on non-IQ quants —
see the quantization table below. The launch profiles are split by placement:
`start-deepseek-mxfp4.sh` (lossless reference, dual + RAM, 256k default),
`start-deepseek-nvidia.sh` (single-card fits, plus the verified RAM-assisted
IQ3 configs), `start-deepseek-nvidia-amd.sh` (all-VRAM dual, Q2_K_XL). None of
this indicts dual-vendor as such — dense models run the same dual runtimes
without a hiccup; it is a backend bug exposed by brand-new model support,
worth retesting after upstream updates.

### Quantization comparison — and the fault isolated to i-quants

Same rig, `-c 131072`, expert-offload dual = `-ts 0,1` with 8 expert layers on
the AMD card (`-ot 'blk\.(3[5-9]|4[0-2])\.ffn_.*_exps.*=ROCm0'`) plus
`--n-cpu-moe` for the remainder:

| Quant | Size | Config | pp t/s | tg t/s | Stability |
|---|---:|---|---:|---:|---|
| IQ3_XXS | 98 GB | cuda + ncmoe 8 | 936–986 | 29.3 | OK (256k verified) |
| IQ3_XXS | 98 GB | dual `-ot` | — | 57.0 | **HIP fault** at 43k–225k |
| MXFP4 | 146 GB | cuda + ncmoe 18 | 299–311 | 16.4 | OK |
| MXFP4 | 146 GB | dual `-ot` + ncmoe 10 | 480–592 | 21.2–24.6 | **OK** — full gauntlet |
| Q8_K_XL | 151 GB | dual `-ot` + ncmoe 11 | 432–557 | 18.6–21.5 | **OK** — full gauntlet |
| Q4_K_XL | 145 GB | dual `-ot` + ncmoe 10 | 489–598 | 21.5–25.1 | **OK** — full gauntlet |

Each dual gauntlet pushes 165k+ tokens of prefill work through the AMD expert
path (4k + 16k bench plus two 65k probes with a full-cache clear between
them). MXFP4, Q8_K_XL and Q4_K_XL all survived it without a fault — roughly
half a million tokens of AMD expert work across three non-IQ quants — while
IQ3_XXS faulted at 43k in one run. That isolates the ROCm fault to the **HIP i-quant
(IQ-series) MoE kernels** on gfx1201 — MXFP4 and k-quant expert paths are
stable, and the expert-offload dual layout is safe (and clearly better than
CPU offload) for those quants.

Practical reading: for quants that fit entirely on the NVIDIA card (IQ3_XXS),
CUDA-only remains the choice. For the larger quants that cannot fit, the
expert-offload dual beats CPU offload by ~1.5–1.9x on prefill and ~1.3–1.5x on
generation, and is stable on non-IQ quants.

**Quality ranking is inverted for this model.** DS4 Flash is
quantization-aware-trained: the routed experts (96% of the model) ship
natively in MXFP4 at ~4.25 bpw, the rest in FP8/BF16. The MXFP4 GGUF repacks
those experts bit-for-bit and is therefore **lossless** — the reference, not a
quantization. Q8_K_XL re-encodes the same FP4 values into a wider format
(more bytes, zero added information) and is dominated by MXFP4 on every axis
here: equal-at-best quality, slightly slower, 5 GB larger. Q4_K_XL re-encodes
the FP4 grid into a different 4-bit grid, which is lossy, and quantizes the
non-expert tensors harder — the only one of the three below the reference.
IQ3_XXS sits lowest; QAT models lose accuracy faster than BF16-trained ones
once experts drop below ~3 bpw.

Q4_K_XL confirmed the prediction empirically: at 145 GB it is the same size
class as MXFP4 (the ~4.25 bpw native experts cannot shrink at "Q4"), needs the
same placement, and measures identically within noise. Being also the lossy
one of the pair, it is dominated by MXFP4 on every axis. **For this model,
keep MXFP4 and skip Q4_K_XL and Q8_K_XL entirely** — the QAT release collapses
the usual quant ladder into a single sensible choice per size class: MXFP4
(reference quality, needs dual or CPU offload) or IQ3_XXS (fits one card,
fastest, lowest quality).

**Related: [antirez/ds4](https://github.com/antirez/ds4)** (DwarfStar) is a
dedicated DeepSeek V4 engine with Metal/CUDA/ROCm backends. Evaluated
2026-08-09: it does not help with this fault — different engine, its ROCm
build targets Strix Halo (`ROCM_ARCH ?= gfx1151`, no gfx1201 mention in the
repo), no mixed-vendor mode (tensor parallel is CUDA-pairs only), and it loads
only its own quant recipes (IQ2_XXS experts / Q4_K / MXFP4 — not IQ3_XXS). Two
things keep it on the radar: TCP pipeline parallelism could in principle chain
a CUDA process and a ROCm process as separate stages (prefill gains only), and
its first-class Strix Halo support with SSD expert streaming makes it a strong
candidate for the planned `halo-linux` rig.

### RAM swap: 4 DIMMs @ 6267 -> 2 DIMMs @ 7400 MT/s

Streaming-read bandwidth (`linux/scripts/membw.c`, 16 threads) went
74.6 -> 89.9 GB/s (+20.5%) — more than the clock ratio, because dropping the
mixed second kit also freed the subtimings. Capacity went 215 -> 122 GiB.

Effect on the RAM-offload configurations, warm state (control: Qwen CUDA d=0
moved ±0.4%):

| Config | metric | before | after | gain |
|---|---|---:|---:|---:|
| MXFP4 dual 128k | tg @ 16k | 22.8 | **23.5** | +3.1% |
| MXFP4 dual 128k | tg @ 65k | 22.5 | 22.9 | +1.8% |
| MXFP4 dual 128k | pp @ 16k | 592 | **613** | +3.5% |
| MXFP4 dual 128k | pp @ 65k | 562 | **586** | +4.3% |
| MXFP4 dual 256k | pp @ 256k | 386 | 389 | +0.8% |
| MXFP4 dual 256k | tg @ 256k | 18.4 | **18.8** | +2.2% |
| IQ3 CUDA-only | tg 4k–65k | 27.7–29.7 | 28.1–29.7 | noise |

**Net: ~+2.5–3% for MXFP4** from a +20.5% bandwidth change — exactly what
Amdahl allows, since only the ~20–25% CPU-expert share of token time is
priced against RAM. Nothing measurable on IQ3, whose CPU share is small. To
extract more from RAM speed the lever is a smaller RAM share (second GPU, or
a smaller quant), not faster RAM.

The capacity cost shows up exactly once per model load: with 122 GiB the
146 GB MXFP4 model cannot stay page-cached, so the **first** prefill after a
load pays NVMe page-in for the CPU-hosted experts (4k pp read 254 vs 480
before — a cold-start artifact, not a regression; 16k+ numbers above are the
steady state).
