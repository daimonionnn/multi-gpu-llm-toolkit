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

| Config | Backend / devices | pp512 (t/s) | vs CUDA | tg128 (t/s) | vs CUDA |
|---|---|---:|---:|---:|---:|
| **cuda** | CUDA0 (NVIDIA alone) | **2232.6** ±28.5 | 100% | **61.17** ±0.06 | 100% |
| vulkan (NVIDIA) | Vulkan1 (NVIDIA alone) | 2168.5 ±11.0 | 97% | 59.47 ±0.20 | 97% |
| rocm-cuda | ROCm0 + CUDA0 | 1635.6 ±35.9 | 73% | 50.58 ±0.05 | 83% |
| vulkan-vulkan | Vulkan2 + Vulkan1 | 1514.2 ±6.8 | 68% | 42.66 ±0.14 | 70% |
| vulkan-cuda | Vulkan2 + CUDA0 | 1506.3 ±41.8 | 67% | 49.56 ±0.07 | 81% |
| vulkan (AMD) | Vulkan2 (AMD alone) | 869.9 ±1.2 | 39% | 25.37 ±0.01 | 41% |
| rocm (AMD) | ROCm0 (AMD alone) | 743.5 ±7.0 | 33% | 24.18 ±0.02 | 40% |

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

| Config | d=0 | d=4096 | d=16384 | d=32768 | retained |
|---|---:|---:|---:|---:|---:|
| vulkan (NVIDIA) | 61.2 | 60.2 | 58.0 | **55.3** | 90% |
| vulkan-vulkan | 42.8 | 42.1 | 40.3 | 37.2 | 87% |
| vulkan (AMD) | 25.4 | 25.0 | 24.1 | 23.1 | 91% |
| rocm (AMD) | 24.2 | 24.0 | 23.3 | 22.3 | 92% |
| rocm-cuda | 50.6 | 49.2 | 21.1 | 13.4 | 26% |
| cuda (NVIDIA) | 64.7 | 62.4 | 21.3 | 12.9 | **20%** |
| vulkan-cuda | 49.6 | 47.4 | 19.4 | 12.1 | 24% |

Prompt processing degrades in the same shape: at 32k, `vulkan (NVIDIA)` holds
941.5 t/s (41% of its depth-0 rate) against `cuda` at 205.7 (9%).

**Everything containing CUDA collapses; nothing else does.** That is the whole
story of this table, and it is a llama.cpp bug rather than a property of the
hardware -- [cuda-fa-blackwell.md](cuda-fa-blackwell.md) has the cause, the
exact 8192 threshold, the power measurements and the fix.

Two things in this table are *not* about that bug and are worth keeping:

- **The ranking inverts between short and long context.** At depth 0 `rocm-cuda`
  is the best dual configuration; by 32k `vulkan-vulkan` is roughly 3x better.
  Backend choices made on depth-0 numbers are wrong for long-context work.
- **Vulkan and ROCm both degrade gracefully**, holding 87-92% of generation
  throughput out to 32k on either vendor.

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
