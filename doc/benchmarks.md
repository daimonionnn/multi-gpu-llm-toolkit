# Benchmarks

For *why* these numbers come out the way they do — the mechanisms behind prefill
and generation, and how to predict a configuration before measuring it — see
[performance-model.md](performance-model.md). This file is the evidence; that
one is the theory.

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

## Results: `halo-win` — DeepSeek V4 Flash MXFP4 (Strix Halo iGPU + RTX PRO 6000, Windows)

Measured 2026-08-15, the day the RTX 5090 was replaced by an RTX PRO 6000. Same
model, same quant and the same 96 GB card as the `dual-linux` section below, so
the two are unusually comparable — except for the link: a full-width CPU slot
there, **four external lanes** here (~8 GB/s). That one difference decides the
whole tuning, in both directions.

> An earlier revision called the `dual-linux` link "PCIe 5.0 x16". That was the
> slot's nominal capability. With the AMD card populating the second slot the
> board split its CPU lanes and the NVIDIA card actually ran at **gen5 x8** —
> inferred from link arithmetic, then confirmed by measurement in *Rewiring:
> the AMD card off the CPU slot* below.

The rig changed under measurement, which is why the tables carry a hardware
column. The card started the day on a **Thunderbolt 5** tunnel; OCuLink was made
to work later (it needs Resizable BAR *disabled* in BIOS), and the BIOS iGPU
framebuffer was tried at 1 GB and 64 GB. Both end states — **OCuLink, 1 GB
framebuffer** — are what the profile ships with.

Setup: lmstudio-community `DeepSeek-V4-Flash-0731-MXFP4` (145.6 GB, 4 shards),
43 layers, MLA attention. Everything below is `-c 131072 -ngl 99 -fa on
-lm none --jinja` over HTTP, single request, 128 predicted tokens, using
`benchmark-loaded-model.ps1`. Runtimes are upstream **prebuilt b10441** binaries
rather than local builds — see [Assembling runtimes without a compiler](#assembling-runtimes-without-a-compiler).

### Final configuration (OCuLink, 1 GB framebuffer)

| Layout | pp 4k | pp 16k | pp 32k | tg 4k | tg 16k | tg 32k |
|--------|------:|-------:|-------:|------:|-------:|-------:|
| **`rocm-cuda`, 18 expert layers on iGPU** | **481.5** | **497.5** | **487.1** | **35.88** | **36.13** | **34.98** |
| `vulkan-cuda`, 18 expert layers on iGPU | 338.8 | 352.1 | 351.6 | 35.14 | 34.35 | 34.94 |
| CUDA-only, `--n-cpu-moe 18` | 266.4 | 299.1 | 296.1 | 23.29 | 22.93 | 22.93 |

`rocm-cuda` wins on both axes — **+41% prefill and +5% generation** over the
Vulkan dual, and +66% / +58% over CUDA-only. That is the profile default.

### How it got there

| # | Layout | Link | FB | pp 4k | pp 16k | pp 32k | tg 4k | tg 16k | tg 32k |
|---|--------|---|---|------:|-------:|-------:|------:|-------:|-------:|
| 1 | CUDA-only, `--n-cpu-moe 18` | TB5 | 1 GB | 245.5 | 260.7 | 262.3 | 22.95 | 22.70 | 22.43 |
| 2 | `vulkan-cuda`, 20 expert layers | TB5 | 1 GB | 300.4 | 317.9 | 312.7 | 32.33 | 32.18 | 30.86 |
| 3 | `vulkan-cuda`, 18 expert layers | TB5 | 1 GB | 328.9 | 342.6 | 340.1 | 33.16 | 33.86 | 33.46 |
| 4 | as 3, CUDA 12.4 runtime instead of 13.3 | TB5 | 1 GB | 115.8* | 336.3 | 329.9 | 17.72* | 34.08 | 33.44 |
| 5 | `vulkan-cuda`, 19 expert layers, `-b 8192 -ub 4096` | TB5 | 1 GB | 318.9 | 325.5 | 320.4 | 32.52 | 32.65 | 31.93 |
| 6 | as 3 | TB5 | 64 GB | 184.2 | 188.0 | 186.4 | 29.81 | 31.09 | 29.49 |
| 7 | `rocm-cuda`, 18 expert layers | TB5 | 64 GB | 452.4 | 468.1 | 458.7 | 32.84 | 32.60 | 31.99 |
| 8 | as 7 | **OCuLink** | **1 GB** | 481.5 | 497.5 | 487.1 | 35.88 | 36.13 | 34.98 |
| 9 | as 3 | **OCuLink** | **1 GB** | 338.8 | 352.1 | 351.6 | 35.14 | 34.35 | 34.94 |
| 10 | as 1, unpinned | **OCuLink** | **1 GB** | 266.4 | 299.1 | 296.1 | 23.29 | 22.93 | 22.93 |
| 11 | as 10 + `--no-op-offload` (experts computed on the CPU) | OCuLink | 1 GB | 99.4 | 100.1 | — | 23.85 | 23.68 | — |
| 12 | repeat of 11, identical config | OCuLink | 1 GB | 98.4 | 99.8 | — | 23.64 | 23.44 | — |
| 13 | as 10 with `-t 4 -tb 4` | OCuLink | 1 GB | 277.9 | 305.4 | — | 20.71 | 20.45 | — |

\* row 4's 4k figures are the PTX JIT on the first request: the CUDA 12.4 build
has no `sm_120` cubins. Its later rows are clean.

`-b 4096 -ub 2048` everywhere except row 5. All dual rows use `-ts 0,1` plus
`-ot 'blk\.(N-42)\.ffn_.*_exps.*=<device>'`, i.e. **only expert FFN tensors**
move to the iGPU.

### The link decides everything — and it is the width, not the tunnel

Row 1 is the same configuration that measures **1175 pp / 16.4 tg** on
`dual-linux`. Here it gives 262 pp / 22.4 tg — a quarter of the prefill and 40%
more generation, from the same card and the same flags.

Both halves follow from the hardware:

- **Generation is faster** because the RAM-hosted experts stream from LPDDR5X
  instead of DDR5. Generation on this model tracks expert-read bandwidth almost
  exclusively, so the memory upgrade shows up almost undiluted.
- **Prefill collapses** because `--op-offload` (on by default) ships batched
  expert matmuls to the GPU, which means pushing the expert weights across the
  link on every micro-batch. Eight times less bandwidth, four times less prefill.

An earlier revision of this file blamed the Thunderbolt tunnel for that gap.
**Rows 1 and 10 disprove it**: replacing TB5 with native OCuLink, same lane
count, buys only +9–15% of prefill (262 → 299 at 16k), not a quarter of it back.
Both links deliver ~8 GB/s; the tunnel adds latency and a little overhead, and
that is all it costs. What starves prefill is **four lanes against sixteen**,
and no cabling change fixes that. (One caveat on that pair: the TB5 run had
pinned host memory and the OCuLink run did not — see the pinning trap below —
so the link is worth slightly more than the 15%.)

So the tuning that wins on `dual-linux` — keep experts in RAM, buy prefill back
with a bigger `-ub` — is the wrong shape here. **Nothing may cross the link
during prefill.**

#### …but the CPU is not the alternative, AVX-512 or not

The obvious reply to a starved link is to stop shipping weights and compute in
place: this rig has 16 Zen 5 cores with AVX-512 and LPDDR5X, against the Arrow
Lake in `dual-linux` that has neither. `--no-op-offload` does exactly that
(rows 11 and 12), and it is **three times worse**:

| CUDA-only, `-ncmoe 18`, 16 threads | pp 4k | pp 16k | tg 4k | tg 16k |
|---|---:|---:|---:|---:|
| `--op-offload` (default) | 266.4 | 299.1 | 23.29 | 22.93 |
| `--no-op-offload`, run 1 | 99.4 | 100.1 | 23.85 | 23.68 |
| `--no-op-offload`, run 2 | 98.4 | 99.8 | 23.64 | 23.44 |

Copying 58 GiB of expert weights across four PCIe lanes and computing on the
Blackwell beats computing in place on the CPU by 3x. Telemetry confirms the role
reversal — during the `--no-op-offload` prefill the CPU sits at 72.9% while the
GPU idles at 13.7% and 81 W. Generation is 3% *better* without op-offload, which
fits: at batch 1 there is nothing to amortise, so the transfer is pure overhead.
`dual-linux` measured the same sign there.

(The two `--no-op-offload` rows were meant to be a 16 vs 32 thread comparison.
They are not: `-t` was passed through `Start-Process -ArgumentList` and never
reached `llama-server`, which logged `n_threads = 16` in both. They stand as a
repeatability check — 1% — and the thread question is answered by row 13
instead, launched directly.)

#### Threads: nothing on prefill, 11% on generation

Row 13 against row 10, `-t 4 -tb 4` against the default 16, verified in the log:

| Threads | pp 4k | pp 16k | tg 4k | tg 16k |
|---:|---:|---:|---:|---:|
| 4 | 277.9 | 305.4 | 20.71 | 20.45 |
| 16 | 266.4 | 299.1 | 23.29 | 22.93 |

**Prefill is unchanged** (4 threads is even marginally ahead, within noise) —
the same result `dual-linux` got, and for the same reason: with `--op-offload`
the host threads have almost nothing to do during prefill. Generation gains
**11%**, more than the 2–4.5% measured on `dual-linux`, but the same shape.

This matters for one cross-rig claim that was overstated in an earlier revision
of this file. `halo-win` generating at 22.9 against `dual-linux`'s 16.4-17.2 in
the identical layout was attributed to LPDDR5X against DDR5. Two measurements
later that is not supportable:

- the `dual-linux` figure was taken at **4 threads** (its profiles never set
  `-t`), and this rig at 4 threads gives 20.45, so part of the gap is thread
  count, not memory;
- `windows/scripts/membw.c`, a port of the Linux benchmark with identical buffer
  size and repeat count, measures **83.8 GB/s at 16 threads here against 89.9
  GB/s** on `dual-linux`. By that benchmark this rig's memory is *slower*.

What survives is a ~22% generation advantage at equal threads whose cause is
**not established**. Candidates: access latency rather than streaming bandwidth
(expert reads are scattered, not sequential), AVX-512 helping the MXFP4 unpack
at batch 1, or the different llama.cpp builds. Not worth guessing about in a
file of measurements.

The ranking that matters is therefore not about the link but about **which
processor does the expert matmuls**: a second GPU that owns the memory (497 pp)
beats shipping to the big GPU (299 pp) beats the CPU (100 pp).

### The layout that follows: zero CPU offload

The iGPU's memory *is* system RAM, so putting the overflow experts there costs
no bus traffic at all. With 18 layers' experts on the iGPU and everything else
on CUDA0, the 146 GB model becomes **fully GPU-resident** — ~89 GB on the
NVIDIA card, ~61 GB on the iGPU, `--n-cpu-moe` unused — and gains 31% prefill
and 41% generation over row 1. Load takes 65–85 s.

The split is the strongest lever available, and it is bounded by CUDA0's VRAM:

| Expert layers on iGPU | CUDA0 used | pp 32k | tg 32k |
|---:|---:|---:|---:|
| 20 | 88 973 MiB | 312.7 | 30.86 |
| 18 | 95 490 MiB | 340.1 | 33.46 |
| 17 | — | does not load at `-c 131072` | |

Each layer moved back to the fast card is worth ~4.5% prefill and ~2%
generation. 18 is the ceiling at 128k context.

### `-ub` is not a lever here — the opposite of `dual-linux`

Row 5 raises the micro-batch to 4096 and loses on both axes. On `dual-linux`
`-ub 2048` was worth +55–60% of prefill, because a larger micro-batch amortises
each transfer of an expert's weights over more tokens. Here **nothing is
transferred** — every expert already sits in the memory of the device that
computes it — so there is nothing to amortise, and the larger compute buffers
cost an expert layer on CUDA0, which is a measurable loss. Spend spare VRAM on
expert layers, not on batch size.

### Only experts may leave the NVIDIA card

A classic layer split (`-ts`, no `-ot`) fails outright. `-ts 19,24` dies with
`CUDA error: out of memory` — the whole KV cache and the compute buffers land
on CUDA0 regardless of the layer proportion. `-ts 22,21` loads and then exits
during warmup, after logging:

```
resolve_fused_ops: layer 2 is assigned to device Vulkan0 but Lightning Indexer
                   is assigned to device CPU (usually due to missing support)
resolve_fused_ops: Lightning Indexer not supported, set to disabled
resolve_fused_ops: fused DeepSeek V4 HC pre / HC comb / HC post not supported,
                   set to disabled
```

Whole layers on the iGPU bring attention with them, and the Vulkan backend
implements none of DeepSeek V4's four fused ops, so llama.cpp disables all of
them globally. **None of those warnings appear in the `-ot` layout**, where
attention stays on CUDA0 and only `ffn_*_exps` tensors move. Keep it that way.

### No Blackwell FA collapse on this model

Rows 3 and 4 are the same layout on CUDA 13.3 and CUDA 12.4. They agree within
noise, and generation is flat from 4k to 32k in both. Neither the 8192 cliff nor
the toolkit dependency documented in [cuda-fa-blackwell.md](cuda-fa-blackwell.md)
appears here — consistent with that bug living in the `fattn` vector/MMA
heuristic for GQA models, which DeepSeek's MLA path does not use.

Practical consequence: **this model needs no CUDA 12.x build and no patch.**
Use the 13.3 prebuilt, which has native `sm_120` cubins and does not pay JIT on
the first request. Nothing here says anything about Qwen or Hermes on this rig;
those have to be measured separately.

### Framebuffer: backend-dependent, not universally better

Row 6 is row 3 with the BIOS iGPU carve-out raised from 1 GB to 64 GB. It costs
**45% of prefill and 8% of generation**, reproduced across two runs
(184.2/195.0 and 188.0/190.8 pp). Placement was verified correct — 58.4 GB
dedicated + 1.4 GB shared on the iGPU, and llama-server's working set fell from
66 GB to 2.4 GB, so the experts really did move into the carve-out.

Utilisation during the run says what changed:

| Framebuffer | NVIDIA util | NVIDIA power | iGPU util |
|---|---:|---:|---:|
| 1 GB | 33.4% | 164 W | 48.2% |
| 64 GB | 28.1% | 127 W | 33.5% |

Both devices do *less* work per unit time — they are waiting on copies, not
computing. The mechanism is `isLargeBar: 0`: the dedicated carve-out is not
fully CPU-visible, so host writes into it go through a small BAR window and a
staging buffer, while GTT-backed shared memory at 1 GB needs no staging at all.
Prefill takes five times the damage generation does, which fits — prefill moves
2048-token activation tensors across the boundary, generation moves one vector
per token. Resizable BAR does not help: HIP reports `isLargeBar: 0` with ReBAR
on over the tunnel, and OCuLink only enumerates with ReBAR off.

**But this is a Vulkan property, not a hardware one.** Row 7 runs the identical
layout on HIP at the same 64 GB framebuffer and shows no penalty at all. That
made the carve-out look mandatory for HIP — and row 8 shows it is not: HIP is
*faster* at 1 GB, allocating its 57.4 GiB out of GTT without complaint. So
nothing wants the carve-out, and **1 GB is the right setting for every layout
here**. Whatever staging path the Vulkan backend takes into dedicated VRAM, HIP
does not take it, and neither needs the dedicated memory to begin with.

### `rocm-cuda` wins on both axes

Best of each backend on the final hardware (rows 8, 9, 10):

| Layout | pp 16k | tg 16k |
|---|---:|---:|
| **`rocm-cuda`** | **497.5** | **36.13** |
| `vulkan-cuda` | 352.1 | 34.35 |
| CUDA-only | 299.1 | 22.93 |

**+41% prefill and +5% generation** over the Vulkan dual. Over TB5 at a 64 GB
framebuffer the same layout gave 468.1 pp / 32.60 tg, so the move to OCuLink
plus a small framebuffer is worth +6% / +11% (two variables, both favourable).
Depth costs it little: at a 65 380-token prompt it measured 441.8 pp / 31.11 tg
on the TB5 configuration.

Reproducibility on the TB5/64 GB configuration: three independent loads gave
468.1 / 465.7 / 460.1 pp and 32.60 / 32.17 / 32.45 tg at 16k, the last one
launched through `start-deepseek-mxfp4-nvidia-amd.ps1` rather than by hand.

#### Trap: pinned host memory has to be decided per layout

`start-llama-server.ps1` has always set `GGML_CUDA_NO_PINNED=1` for the `rocm`,
`cuda` and `rocm-cuda` modes. **Each setting of that variable breaks one of the
two layouts**, and both failures are loading failures, not slowdowns.

*Dual layouts need pinning ON.* With a 64 GB carve-out, `GGML_CUDA_NO_PINNED=1`
prevents the model from loading at all. Two runs, two symptoms, both stopping at
exactly 89 846 MiB of CUDA VRAM — the point where the single large iGPU buffer
is allocated:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 58752.00 MiB on device 0:
    cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate ROCm0 buffer of size 61605937152
```

and, in the other run, a hard hang at the same point: VRAM frozen, no CPU time
accumulating, no disk I/O. Without the variable the identical command loads in
60–70 s, reproduced three times. The mechanism is UMA plus a large carve-out:
unpinned loading keeps its staging in pageable host memory, and here host memory
and the carve-out are the *same physical RAM* — with 64 GB carved out the OS has
63.6 GB left, and a 57.4 GiB device allocation collides with the staging.

*CUDA-only needs pinning OFF.* Its RAM-hosted experts are one ~58.4 GiB host
buffer, and pinning that much simply fails:

```
alloc_tensor_range: failed to allocate CUDA_Host buffer of size 62664998912
```

with **112.7 GB of RAM free** — it is the size of the single pinned allocation
that fails, not a shortage of memory. The same layout loaded pinned earlier in
the day, so it sits right at the edge and depends on fragmentation. Unpinned it
loads every time, at the cost of slower host-to-device transfer, which is why
row 10 is the number to expect from this layout rather than row 1.

`start-llama-server.ps1` therefore takes `-AllowPinned`, which suppresses the
variable. The default is unchanged, so existing profiles behave as before; the
DeepSeek profile passes it for the dual layouts and withholds it for CUDA-only.

### Prefill: what is left, and what the noise floor is

A round of tuning aimed specifically at prefill, after the layout itself was
settled. **Nothing moved it**, and knowing why is worth more than the attempts:

| Attempt | pp @16k | Verdict |
|---|---:|---|
| reference (18 layers, `-ub 2048`) | 497.5 / 480.1 / 475.4 | three runs of the *same* config |
| Windows power mode | — | no effect; see below |
| `-ub 1024` | 435.7 | −10%, and it frees only 1 GB |
| 17 expert layers on the iGPU | — | needs 3.4 GB free, cannot be reached |
| llama.cpp b10453 | — | not run: 12 commits, none touching deepseek/MLA/FA/HIP |

**The noise floor here is ±2–3%.** Three runs of the identical configuration
span 475–497 pp. Nothing below ~5% should be called an improvement in this file,
and two earlier single-run comparisons were re-read in that light.

The power-mode attempt is also a lesson in verification: `powercfg
/overlaysetactive` returns exit 0 and changes nothing, so the first "Best
performance" measurement was the Balanced setting measured twice. The registry
value `ActiveOverlayAcPowerScheme` is what to check.

`-ub 1024` was tried to free VRAM for a 19th layer on the fast card — spend the
micro-batch, buy a layer. It fails twice over: the micro-batch is worth −10%
while a layer is worth +4.5%, and it frees 1 GB where a layer needs 3.4.

### The prefill lever that is not a flag: prompt structure

The prefix cache is worth more than every flag in this section combined. Same
rig, same config, measured over HTTP with `cache_prompt` (on by default):

| Request | Tokens actually processed | Time |
|---|---:|---:|
| Cold 24 000-token prompt | 24 003 | **52.6 s** |
| Append 1 400 tokens to it | 1 404 | **3.5 s** |
| Change the **first** 10 tokens | 25 415 | **55.8 s** |
| Return to an earlier version of the prompt | 4 | **0.11 s** |

**Appending is 15x cheaper than re-prefilling; changing anything near the start
costs the whole prompt again.** The last row is the four server slots doing
their job — an earlier branch of the conversation was still resident.

So on a rig where prefill is the expensive phase, the highest-value change is in
how prompts are assembled, not in how the server is launched: put the stable
material first (system prompt, tool definitions, retrieved documents) and let
only the tail vary. Editing a system prompt costs a full re-prefill — 53 s here,
against 3.5 s for the same content appended at the end.

`--cache-reuse` does **not** rescue the other cases. Tested at 256 on both a
prepend and a 400-token insertion in the middle of a 24k document, it reused
nothing in either: 55.4 s and 53.1 s, indistinguishable from cold. It is not in
the profile for that reason.

### Stability: the `deepseek4` HIP fault has not appeared here

Two consecutive 65 380-token prefills with a full cache clear between them,
on the `rocm-cuda` layout:

| Probe | pp | tg |
|---|---:|---:|
| 65k #1 | 441.8 | 31.11 |
| 65k #2 (after clear) | 440.8 | 31.09 |

Roughly **184 000 prefill tokens through the AMD expert path, zero
`HSA_STATUS_ERROR_MEMORY_FAULT`**, with throughput identical to the decimal.

This certifies nothing, and the reason is recorded further down this file: on
`dual-linux` a 165k-token gauntlet passed and the same layout died at ~45k
tokens in production hours later. What is genuinely different here is the
hardware — gfx1151 on Windows against gfx1201 on Linux — so the `deepseek4`
HIP bug may simply not apply. Only sustained real use will tell.

### Assembling runtimes without a compiler

No CUDA Toolkit is installed on this rig ([systems.md](systems.md)), so
`setup-llama.ps1` cannot build anything containing CUDA. It was not needed.
Every backend DLL in upstream's b10441 Windows releases is built with
`GGML_BACKEND_DL`, so backends from *the same build* can be mixed by copying
DLLs into one directory — the Windows analogue of what
`linux/scripts/build-cuda12-container.sh` does with a container:

| Runtime | Assembled from | Devices |
|---|---|---|
| `runtime-cuda133` / `runtime-cuda124` | release zip + matching cudart zip | CUDA0 |
| `runtime-vulkan` | `win-vulkan` zip | Vulkan0 (AMD), Vulkan1 (NVIDIA) |
| `runtime-vulkan-cuda133` | CUDA runtime + `ggml-vulkan.dll` | CUDA0, Vulkan0, Vulkan1 |
| `runtime-rocm-cuda133` | CUDA runtime + locally built `ggml-hip.dll` | CUDA0, ROCm0 |

Only the HIP backend had to be compiled, because the upstream ROCm build is ABI
incompatible with the installed SDK (details in [systems.md](systems.md)).
Building just that one DLL, single-arch, takes minutes:

```powershell
# from a shell where vcvars64.bat -vcvars_ver=14.44 has been imported
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER="$env:HIP_PATH/bin/clang.exe" `
    -DCMAKE_CXX_COMPILER="$env:HIP_PATH/bin/clang++.exe" `
    -DGGML_HIP=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF `
    -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF `
    -DAMDGPU_TARGETS=gfx1151 -DGPU_TARGETS=gfx1151
cmake --build build-hip --target ggml-hip -j 14
```

Check out the **same tag** as the prebuilt (`b10441` here) before building, or
the ABI will not match.

### Not yet measured on this rig

- Anything other than DeepSeek V4 Flash. The Blackwell FA finding above is
  model-specific and Qwen/Hermes may well hit the collapse.
- `vulkan-vulkan`, and the `Lucebox/DeepSeek-V4-Flash-ROCMFP2-STRIX` quant
  (95.3 GB) that would nearly fit the NVIDIA card alone.
- Long-run stability of the AMD expert path under real traffic. The soak above
  ran on the TB5 configuration; nothing has been soaked on OCuLink yet.
- Whether the expert split still tops out at 18 layers on OCuLink — the ceiling
  is CUDA0's VRAM, which did not change, so it should, but 17 was never retried.
- A wider link. Four lanes is the binding constraint on prefill for any layout
  that moves weights; a gen4 or gen5 x16 slot would be the one upgrade that
  changes the shape of these results.

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
`start-deepseek-mxfp4-nvidia-amd-cpu.sh` (lossless reference, dual + RAM, 128k default -
faster than 256k because the smaller KV cache keeps two more expert layers
off DDR5; `--256k` for the full window),
`start-deepseek-iq2xxs-nvidia.sh` (single-card fits, plus the verified RAM-assisted
IQ3 configs), `start-deepseek-q2kxl-nvidia-amd.sh` (all-VRAM dual, Q2_K_XL). None of
this indicts dual-vendor as such — dense models run the same dual runtimes
without a hiccup; it is a backend bug exposed by brand-new model support,
worth retesting after upstream updates.

### Quantization comparison — and the fault isolated to i-quants

Same rig, `-c 131072`, expert-offload dual = `-ts 0,1` with 8 expert layers on
the AMD card (`-ot 'blk\.(3[5-9]|4[0-2])\.ffn_.*_exps.*=ROCm0'`) plus
`--n-cpu-moe` for the remainder:

| Quant | Size | Config | pp t/s | tg t/s | Stability |
|---|---:|---|---:|---:|---|
| IQ3_XXS | 98 GB | cuda + ncmoe 8 | 936–986 | 29.3 | OK (256k verified; profile: `start-deepseek-iq3xxs-nvidia-cpu.sh`) |
| IQ3_XXS | 98 GB | dual `-ot` | — | 57.0 | **HIP fault** at 43k–225k (profile: `start-deepseek-iq3xxs-nvidia-amd.sh`) |
| MXFP4 | 146 GB | cuda + ncmoe 18 | 299–311 | 16.4 | OK |
| MXFP4 | 146 GB | dual `-ot` + ncmoe 10 | 480–592 | 21.2–24.6 | **OK** — full gauntlet |
| Q8_K_XL | 151 GB | dual `-ot` + ncmoe 11 | 432–557 | 18.6–21.5 | **OK** — full gauntlet |
| Q4_K_XL | 145 GB | dual `-ot` + ncmoe 10 | 489–598 | 21.5–25.1 | **OK** — full gauntlet |

Each dual gauntlet pushes 165k+ tokens of prefill work through the AMD expert
path (4k + 16k bench plus two 65k probes with a full-cache clear between
them). MXFP4, Q8_K_XL and Q4_K_XL all survived it without a fault — roughly
half a million tokens of AMD expert work across three non-IQ quants — while
> **Production correction (2026-08-09, evening).** The i-quant isolation below
> did not hold. Serving a real agent workload, the **MXFP4** dual faulted with
> the same `HSA_STATUS_ERROR_MEMORY_FAULT` at ~45k prefill tokens — hours
> after passing its gauntlet. The bug is **probabilistic on the AMD expert
> path for all quants**, IQ quants merely failing fastest; a 165k-token
> gauntlet cannot certify a low-rate fault. Unattended/fallback duty must run
> CUDA-only (`start-deepseek-mxfp4-nvidia-cpu.sh`).
>
> **Scope narrowed (2026-08-10): it is deepseek4-specific.** gpt-oss-120b
> (`gpt-oss` arch, 128 experts, native MXFP4) ran the experts of 17 of its 36
> layers on the AMD card and pushed **1,008,000 prefill tokens** through that
> path — 8 × 126k probes with full-cache clears — at a rock-steady 2060 pp /
> 85 tg, zero faults. That is 4x more AMD expert work than DeepSeek ever
> survived, so the fault lives in the `deepseek4` compute path (days old in
> llama.cpp), not in MoE-on-RDNA4 generally. **Dual layouts are back on the
> table for non-DeepSeek MoE** (Step-3.7-Flash class); DeepSeek stays off the
> AMD card until upstream fixes land.

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

#### Post-update check (2026-08-12, build 10429 / `77918caf3`)

Re-measured after pulling 98 upstream commits, to confirm the update cost
nothing. Same rig, MXFP4 expert-offload dual, single stream:

| Context | `-ncmoe` | pp t/s | tg t/s |
|--------:|---------:|-------:|-------:|
| 4 096 | 10 | 532.0 | 23.8 |
| 16 384 | 10 | 597.8 | 22.9 |
| 32 768 | 10 | 591.5 | 22.8 |
| 131 072 | 10 | 512.2 | 21.7 |
| 262 144 | 12 | 387.4 | 18.8 |

Against the recorded baselines — 480-592 pp / 21.2-24.6 tg at 128k and
386 pp / 18.4 tg at 256k — this is at or just above the range at 128k and
indistinguishable at 256k. **No regression.** The small gain at 128k must not
be credited to the update: `--no-mmap` was added in the same window, so two
variables moved at once.

Generation falls only 23.8 -> 21.7 across a 32x context increase, which is MLA
doing its job: the compressed KV cache grows slowly enough that depth barely
touches the bandwidth-bound decode.

**445 725 tokens of prefill through the AMD expert path, zero HSA faults** —
and that still does not mean the fault is gone. The previous gauntlet was 165k
tokens, passed, and the same layout died at ~45k in production hours later. A
clean run moves the estimate of a low-rate probabilistic fault very little.
What it does establish is that `ebb546b7e`, which widens CUDA-graph use onto
the MoE path we fault on, did not break anything outright.

#### Which MXFP4 profile — and is the AMD card still worth having?

Head to head on 2026-08-12, both profiles on `-b 4096 -ub 2048`, back to back,
single stream:

| Context | cuda-only pp | dual pp | cuda-only tg | dual tg |
|--------:|-------------:|--------:|-------------:|--------:|
| 4 096 | **1096.6** | 884.3 | 16.81 | **23.28** |
| 16 384 | **1172.6** | 948.1 | 17.23 | **22.98** |
| 32 768 | **1131.5** | *faulted* | 16.97 | *faulted* |

**They disagree, and the disagreement is the whole answer.** CUDA-only
prefills ~24% faster; the dual generates ~35% faster. Both follow from where
the expert weights are. Generation is bandwidth-bound — each token reads its
active experts, and the AMD card serves them from VRAM at ~640 GB/s against
DDR5's ~75 GB/s, which is worth 23 t/s versus 17. Prefill is now GPU-side work
(`--op-offload`), where coordinating two backends costs the dual more than the
second card returns.

So the answer to "can the AMD card go" is **no, but the case for it is
narrower than it looks on this model**:

- It buys **+35% generation** on DeepSeek MXFP4 — the metric felt token by
  token, where prefill is a one-off wait.
- **Step-3.7-Flash Q4_K_S does not fit without it.** 104 GB against 96.6 GB of
  NVIDIA VRAM; with both cards it runs entirely in VRAM at 2100-2440 pp /
  78-94 tg.
- Against that: **every fault on this rig is on the AMD path.** Five today,
  all on dual profiles, none on CUDA-only across at least as much work.

For anything unattended — the hermes fallback above all — CUDA-only is now
strictly the right choice, and no longer a compromise: since `-ub 2048` it is
also the faster of the two at prefill.

#### `-ub` is the prefill lever: +60% for one flag

The mainline profiles never set `-b`/`-ub` either, so they ran on llama.cpp's
`-b 2048 -ub 512`. Interleaved arms, `-t 24 -tb 24` throughout, contexts
4k/16k/32k:

| `-b`/`-ub` | pp 4k | pp 16k | pp 32k | tg 4k | notes |
|-----------|------:|-------:|-------:|------:|-------|
| 2048 / 512 (default) | 565 | 595 | 597 | 25.5 | |
| 4096 / 1024 | 833 | 891 | 887 | 25.0 | |
| **4096 / 2048** | **937** | **947** | **907** | 24.6 | **best; clean in both rounds** |
| 4096 / 3072 | 898 | — | — | 24.1 | run ended in the HSA fault |
| 8192 / 4096 | 964 | **OOM** | — | 24.3 | loads, then cannot allocate at 16k |
| 8192 / 8192 | — | — | — | — | will not load |
| `--no-op-offload` | 282 | 278 | 272 | 26.3 | diagnostic |

**`-ub 2048` is worth ~+55-60% of prefill and costs nothing in generation.**
Bigger micro-batches let more tokens share one load of an expert's weights,
which is the dominant cost when experts live outside the GPU. The ceiling is
between 2048 and 4096: `-ub 4096` starts, serves 4k, then dies of
`cuMemCreate ... out of memory` at 16k, because its compute buffers no longer
fit beside the expert layers at `--n-cpu-moe 10`.

Carried into the other modes and verified at 16k, since `-ub 4096` had already
shown that a config can load cleanly and still OOM once a real context is
allocated:

| Mode | `-ncmoe` | pp with `-ub 2048` | previously |
|------|---------:|-------------------:|-----------:|
| default (128k, dual) | 10 | 947 | 595 |
| `--256k` (dual) | 12 | 839 | — |
| **`--cuda-only`** | 18 | **1175** | **~305** |

The CUDA-only profile gains **almost 4x**, far more than the dual — and that
follows from the mechanism rather than contradicting it. It keeps 18 expert
layers in system RAM against the dual's 10, so a larger share of its work is
weight transfer, and a larger share is therefore amortised by the bigger
micro-batch. This is the profile the systemd fallback runs: a 63.5k-token
hermes conversation now prefills in ~54s instead of ~208s, which moves that
fallback from marginal to comfortable.

The `--no-op-offload` arm explains the shape of everything else. Prefill falls
to ~275 — **less than half the default and under a third of the tuned
figure** — because the batched expert matmuls then run on the CPU instead of
being shipped to the GPU. That is why threads were worth nothing on prefill
(§ above) and why `-ub` is worth so much: with op-offload on, the micro-batch
size sets how well each weight transfer is amortised. Generation is very
slightly *better* without it (26.3 vs 25.5), which fits — at batch size 1
there is nothing to amortise, so the transfer is pure overhead.

**This invalidates the mainline side of the ik_llama comparison.** That
head-to-head gave mainline ~305 pp against ik's 385 and concluded ik wins
prefill by 26% — but mainline ran there on `-ub 512`, untuned, exactly the
kind of unfair setup that comparison was itself correcting on ik's behalf. It
needs re-running with `-ub 2048` on both sides before the verdict means
anything.

One caveat on the fault column: the `ub4096` row is marked ok because the
detection grep only looked for `HSA_STATUS`, and that run died of a CUDA OOM
instead. Corrected here by hand.

#### Threads: worth ~3% on generation, nothing on prefill, and not a fault trigger

The mainline profiles never set `-t`, so the server had been running on
**4 threads of 24** while 10-12 expert layers were computed on the CPU. On
ik_llama, 12 -> 24 threads was worth +32% prefill, so this looked like a large
missed lever.

It is not, and the first measurement of it was misleading. A single 24-thread
run showed +6.3% pp and +7.1% tg against the 4-thread numbers from earlier the
same day — but those baselines came from a different server instance, and that
run also faulted after 3 949 tokens, which made thread count look like a fault
trigger as well.

Both readings dissolved under an interleaved trial: 3 pairs, alternating 24 and
4 threads, identical workload (4k + 16k + 32k), ~330 000 prefill tokens total.

| Context | pp @24t | pp @4t | Δ | tg @24t | tg @4t | Δ |
|--------:|--------:|-------:|--:|--------:|-------:|--:|
| 4 096 | 564.4 | 561.6 | +0.5% | 25.38 | 24.28 | **+4.5%** |
| 16 384 | 597.6 | 596.1 | +0.2% | 24.94 | 24.01 | **+3.9%** |
| 32 768 | 596.8 | 597.2 | −0.1% | 23.93 | 23.45 | **+2.0%** |

**Prefill is identical** — the +6.3% was cross-instance noise. That confirms
mainline offloads the batched expert matmuls to the GPU (`--op-offload`, on by
default), so host threads have almost nothing to do during prefill. It is also
why mainline prefills at ~597 on 4 threads where ik_llama needs 24 to reach
355: different engines, opposite bottlenecks. **Generation gains 2-4.5%**,
which is the CPU computing active experts at batch size 1, where 4 threads do
not saturate DDR5.

**Threads do not trigger the fault.** Three 24-thread runs completed clean, so
the earlier 3 949-token death was the probabilistic fault doing what it does,
not a thread effect. (One 32k run at 24 threads returned 289.87 pp / 16.92 tg,
roughly half — treated as contention and excluded; every other run of that
cell sits within 1%.)

Worth setting `-t 24 -tb 24` for the ~3% of generation, but it is not the
missing lever, and the remaining candidates are `-b`/`-ub` and `--op-offload`.

#### Upstream fix status (re-checked 2026-08-12) — still none

98 commits later, nothing fixes it. [#26738](https://github.com/ggml-org/llama.cpp/issues/26738)
is still open and its fix [#26771](https://github.com/ggml-org/llama.cpp/pull/26771)
is still unmerged — moot for us either way, see below.

Two commits in that window matter to this configuration, and the first is a
**risk rather than a fix**:

- `ebb546b7e` "CUDA: only disable CUDA graphs when mul_mat_id actually needs a
  stream sync" narrows when graphs are switched off around MoE expert matmuls,
  so graphs are now used in *more* cases on exactly the code path where we
  fault. Whether that helps or hurts here is unmeasured.
- `e79e4bf66` drops `-funsafe-math-optimizations` from HIP builds, because
  `-fassociative-math` reassociates FP reductions and can flip argmax. A
  correctness fix, not a crash fix, but it changes what runs on the AMD card.

Also worth knowing, though **we are not affected**:
[#26399](https://github.com/ggml-org/llama.cpp/issues/26399) reports
`GGML_OP_TOP_K` falling back to CPU on HIP above ~3-4k context, costing 6.4x
of token generation (15.78 -> 2.47 t/s) on this very model and architecture.
That reporter runs DeepSeek fully GPU-resident across six MI50s, so routing
runs on AMD. Our `-ot` layout keeps only expert *weights* on the AMD card
while attention and the router stay on CUDA0, and the measurements agree:
21.2-24.6 tg on the dual, and 22.9 tg at 12k context in production, with no
discontinuity past 4k.

#### Upstream fix status (checked 2026-08-10) — still none

22 commits landed upstream after the build we run; none touches the
`deepseek4` or HIP MoE path. ROCm is 7.1.1 with nothing newer in the
repository, and neither ds4 fork has done any gfx1201 work.

One upstream bug looked like an excellent match and had to be **ruled out**:
[#26738](https://github.com/ggml-org/llama.cpp/issues/26738), fix in the open
PR [#26771](https://github.com/ggml-org/llama.cpp/pull/26771).
`ggml_cuda_pool_leg::clear_pool()` frees scratch buffers that an already
captured HIP graph still points at, so the next replay writes to freed memory.
Its stated preconditions all hold in our build (`GGML_HIP_NO_VMM=ON`,
`GGML_HIP_GRAPHS=ON`, MoE expert offload), and it would have explained the
fault far better than architecture does — under it the trigger is VRAM
pressure, so DeepSeek faults where the smaller gpt-oss does not, with no need
for anything `deepseek4`-specific.

It is not our bug. `clear_pool()` only runs when an allocation actually fails,
and **with `-fa on` it never does**: flash attention never materialises the KQ
matrix, so buffer demand reaches steady state instead of growing with context
depth. That is why the reporter needed `-fa 0` — the condition is causal, not
incidental to their setup.

Measured rather than assumed. Two runs at `-lv 5` (the flush message is
`GGML_LOG_DEBUG`, which the default INFO verbosity discards — so no earlier
crash log of ours could ever have shown it):

| run | workload | AMD VRAM free | pool flushes | fault |
|---|---|---:|---:|---|
| 8 expert layers | 4 x 65k identical prefills | 3.8 GiB | 0 | no |
| 9 expert layers | 385k tokens, 3k-65k prompts, mixed cache/generation | **0.6 GiB** | **0** | no |

CUDA graphs were demonstrably live throughout (2,627 graph warmups in the
second run), so the absence of flushes is not an absence of graphs. Deliberate
starvation to 600 MiB free did not produce a single allocation failure.

So the fault remains unexplained and unfixed, and **no upstream issue
describes our case**. Note also that this run is not evidence of stability:
385k clean tokens is the same kind of pass the MXFP4 dual gave before faulting
in production hours later.

Two other open issues do match observations recorded here:
[#25664](https://github.com/ggml-org/llama.cpp/issues/25664) is our Vulkan
`DeviceLostError` on DeepSeek V4 Flash, and
[#26220](https://github.com/ggml-org/llama.cpp/issues/26220) reports up to 2x
prefill loss on RDNA4 after the rocWMMA removal.

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

### Rewiring: the AMD card off the CPU slot, 2026-08-17

The R9700 was moved out of the second CPU slot and now hangs off a switched
path on chipset root port `1d.4`, so the RTX PRO 6000 has the CPU lanes to
itself. Both ends of that trade are measurable, and they point in opposite
directions.

The AMD card still enumerates normally — `ROCm0`, 32 GB, gfx1201, and it loads
expert layers as before. What changed is the link. Reading each hop of the
chain from sysfs:

```
1d.4-[b0-b2]--00.0-[b1-b2]--00.0-[b2]--00.0  Radeon AI PRO R9700
       ^ x4 @ 16.0 GT/s          x16 @ 32 GT/s from here on
```

The device negotiates x16 gen5 with the last bridge, so `current_link_width`
read at the card alone says x16 and means nothing. The binding hop is the
first one: **PCIe 4.0 x4, ~7.9 GB/s**. The NVIDIA card reads x16 with a gen5
ceiling on CPU root port `06.0`.

Measured with the same harness as the `-ub` table above — HTTP, single
request, 128 predicted tokens, `-c 131072`, `-b 4096 -ub 2048` verified on the
running process rather than assumed:

| Profile | metric | before | after | Δ |
|---|---|---:|---:|---:|
| `--cuda-only` (`ncmoe 18`) | pp 4k | 1175 | **1399** | **+19%** |
| | pp 16k | 1175 | **1532** | **+30%** |
| | tg | 16.4 | **17.5–18.2** | **+6–11%** |
| default dual (`ncmoe 10` + `-ot`) | pp 4k | 937 | **561** | **−40%** |
| | pp 16k | 947 | **615** | **−35%** |
| | tg | 24.6 | 22.8–23.3 | −6–7% |

**The profile that never touches the AMD card got faster, and the one that
does got slower.** Both follow from the same rewiring. CUDA-only keeps 18
expert layers in system RAM and streams them to the GPU, so host-to-GPU
bandwidth is its prefill bottleneck — widening that link is worth 19-30%. The
dual keeps expert layers 35-42 in AMD VRAM, and every prefill batch now pushes
their activations through four lanes. Generation loses only ~6% because a
single token moves far less across the link than a 2048-token micro-batch.

This inverts the recommendation in *Which MXFP4 profile* above on the one axis
where the dual still won. The dual was already off the table for unattended
duty because of the HSA fault; it now loses prefill by a factor of 2.5 as
well, and keeps only its generation lead.

**This settles the x8 question.** [systems.md](systems.md) and
[performance-model.md](performance-model.md) had already argued from link
arithmetic that the NVIDIA card was running at **gen5 x8**, not the x16 the
inventory recorded, because the board splits its CPU lanes when the second
slot is populated. Confirmed: freeing those lanes bought 30% of prefill,
which is impossible if the card already had x16. Implied link throughput in
the CUDA-only layout moves accordingly — 62.66 GB per batch over 4096/1532 s
is **23.4 GB/s**, against 18.0 GB/s before.

**Both standing predictions were half right.** *Still open* in systems.md
proposed exactly this rewiring and forecast two things:

| Prediction | Outcome |
|---|---|
| Restoring x16 raises CUDA-only prefill to ~1800–2400 pp | **Over-forecast.** Measured 1532. At 23.4 GB/s the link is only 37% utilised, so it is no longer the binding constraint — the hedge that "the bottleneck moves elsewhere" was right, and it moves sooner than the arithmetic suggested. |
| A chipset x4 slot costs the dual only single-digit percent, since it carries ~1 GB of activations per micro-batch rather than 62.66 GB of weights | **Wrong.** The dual lost 35–40% of prefill. Activation volume alone does not predict the cost; the four-lane hop sits in the critical path of every micro-batch, and latency there is not amortised the way a bulk weight transfer is. |

The second miss is the more useful one: it means "the dual barely uses the
link" is not a safe assumption when deciding where to put a second card.

**Caveat on precision.** Each cell here is a single run on a fresh server
instance, not the interleaved arms used for the `-ub` and thread tables. The
threads section above is a standing warning about exactly this: a
cross-instance comparison there manufactured a +6.3% that dissolved under
interleaving. The directions and magnitudes are far outside that noise floor —
35-40% against ~1% — but the individual percentages deserve a proper
interleaved re-run before being quoted.

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

### DeepSeek all-VRAM 2-bit quants — the single-card speed class

Two ~2-bit quants small enough to fit the NVIDIA card entirely (no RAM
offload, no AMD), `-c 131072`, measured 2026-08-10:

| Quant | Size | pp 4k/16k/65k | tg 4k/16k/65k |
|---|---:|---|---|
| unsloth UD-IQ2_M | 85 GB | 2118 / 2106 / 1620 | **76.1 / 75.1 / 65.2** |
| antirez IQ2XXS+Q8-attn (imatrix) | 80.8 GB | 2007 / 2040 / 1557 | 74.5 / 73.4 / 63.9 |

2.5x the generation of the previous best DeepSeek config (IQ3+ncmoe8:
29.3 tg) — eliminating the last RAM-hosted expert layers is worth far more
than the raw bandwidth math suggests.

Speed is a tie within noise, so **quality decides, and it favours antirez**
(published numbers against a Q8 reference, not measured here):

| Quant | Size | PPL | Mean KLD | Same top token |
|---|---:|---:|---:|---:|
| **antirez IQ2XXS+Q8-attn (imatrix)** | 86.7 | **6.0808** | **0.4079** | **78.15%** |
| unsloth UD-IQ2_M | 85 | 7.0894 | 0.4839 | 76.56% |
| *(context)* Q2_K_XL | 90.2 | 6.6782 | 0.4077 | 78.57% |
| *(context)* IQ3_XXS | 97.1 | 6.1972 | 0.3079 | 81.93% |

Better on every axis while being smaller. The asymmetric recipe is why:
routed experts go to IQ2_XXS/Q2_K but attention projections, shared experts,
router, output head and embeddings stay Q8/F16 — the per-token decision
machinery keeps its precision while the experts, each seeing a fraction of
tokens, absorb the loss. Its imatrix is calibrated on chat-v2 traffic, which
the author says specifically restores tool-calling and instruction-following
quality. That makes it the default for `start-deepseek-iq2xxs-nvidia.sh`, and the
right pick for agent duty among the 2-bit options.

Still, all of these are ~2-bit re-encodes of a QAT model whose experts are
natively ~4.25 bpw: ~1 token in 5 differs from the Q8 reference. They are the
speed picks; MXFP4 remains the quality pick, and gpt-oss-120b below offers
full fidelity of a smaller model at 3x the speed.

Sources: [Unsloth DeepSeek-V4 docs](https://unsloth.ai/docs/models/deepseek-v4),
[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf).

### The RAM-offload cliff: why 3-bit DeepSeek is not worth it here

IQ3_S (108 GB) looked like the sweet spot between the 2-bit speed picks and
MXFP4 — better quality, only a little too big for the card. It is not:

| Quant | Size | RAM-hosted expert layers | pp 4k/16k/65k | tg |
|---|---:|---:|---|---:|
| antirez IQ2XXS+Q8 | 80.8 GB | 0 | 2007 / 2040 / 1557 | **74.5** |
| IQ3_XXS | 97.1 GB | 8 | 936 / 986 / 876 | 29.3 |
| IQ3_S | 108 GB | 10 | 574 / 790 / 731 | 27.4–28.4 |
| MXFP4 (`--cuda-only`) | 146 GB | 18 | ~305 | 16.4 |

`--n-cpu-moe 4` and `7` both OOM'd; 10 was the first that loaded, and by then
the speed is IQ3_XXS's with none of the size advantage. The pattern across
the whole table is the real lesson: **on this rig generation speed tracks the
number of RAM-hosted expert layers almost linearly, and nothing else.** The
step from 0 layers to 8 costs 60% of throughput — far more than the ~20% the
memory-bandwidth arithmetic predicts, because each RAM layer adds
synchronisation and CPU compute, not just slower reads.

Practical consequence: for DeepSeek there are exactly two sensible points,
not a ladder. Everything that fits the card whole (~2-bit, 74 tg) or the
lossless MXFP4 accepting 16 tg. The 3-bit middle is dominated by both.

### gpt-oss-120b — the fastest model on this rig

Native-MXFP4 MoE (60 GB, 128 experts / 4 active), fits the NVIDIA card whole:

| Config | pp | tg |
|---|---|---|
| CUDA-only, 4k / 16k | 9950 / 9448 | **258 / 242** |
| CUDA-only, 126k prompt | 4655 | 148.8 |
| dual, 17 expert layers on AMD | ~2060 | ~85 |

The dual row is the stability experiment above, not a recommendation — with
the model fitting one card, adding the AMD card costs 3x. CUDA-only gpt-oss
is the speed king of this machine by a wide margin.

### Step-3.7-Flash — the model dual-GPU actually exists for

`step35` arch, 45 layers, 288 experts / 8 active. stepfun-ai Q4_K_S is
**104 GB**: too big for the 96.6 GB NVIDIA card, comfortable inside 96.6 +
32.6 GB of combined VRAM with **no CPU offload at all**. Layout: all layers,
KV and attention on CUDA0 (`-ts 0,1`), experts of 10 layers (~24 GB) on the
AMD card. Context 65536.

| Measurement | pp | tg |
|---|---:|---:|
| 4k / 16k bench | 2194 / 2441 | 89.2 / 85.1 |
| 6 × 64k probes (clears between) | 2074–2143 | 77.6–78.3 |

**384,000 prefill tokens through the AMD expert path, zero faults**, with
throughput flat to the last decimal — the `step35` arch is unaffected by the
`deepseek4` HIP bug, as the gpt-oss result predicted.

Two things worth noting. First, 78–94 tg on a 104 GB model beats DeepSeek's
80 GB all-VRAM 2-bit pick (74.5 tg): with 8 of 288 experts active, Step-3.7
reads less per token than its file size suggests, so MoE sparsity matters
more than model size. Second, this is the first configuration in the project
where the AMD card is **load-bearing rather than optional** — remove it and
the model does not run at this speed at all.

Profile: `start-step37-q4ks-nvidia-amd.sh`. 128k is the default and already
carries the extra expert layer on AMD that the bigger KV cache pays for;
`--64k` is the smaller-window variant.

> **Trap:** unsloth's `UD-Q4_K_XL-R4` (114 GB) will not load in mainline
> llama.cpp — `tensor 'blk.3.ffn_down_exps.weight' has invalid ggml type 213`.
> The `-R4` row-interleaved repack is an **ik_llama.cpp** extension. Use the
> stepfun-ai Q4_K_S for mainline.

### ik_llama.cpp evaluated for the RAM-offload case

`ik_llama.cpp` (fork at `bd342d6`, "DS4 optimizations") is built around exactly
our weakest configuration — large MoE with experts in system RAM — with MLA
variants, `-fmoe`, run-time repacking and R4 quants. Measured against our
mainline MXFP4 CUDA-only, same model, same 128k context:

| Engine | pp 4k/16k/65k | tg 4k/16k/65k |
|---|---|---|
| mainline `--cuda-only` | ~305 | **16.4** |
| ik_llama (`-mla 3 -fidx`) | 214 / 217 / 203 | 19.5 / 19.6 / 17.5 |

That first comparison was unfair on both counts, and re-doing it properly
reverses the verdict. The ik binary on disk predated its own checkout by two
weeks, missing a burst of DS4 optimisations landed 2026-08-07/08, and it ran
with f16 KV and default threads. Current build, q8_0 KV, 24 threads:

| Engine (both CUDA 12.8 class) | pp 4k/16k/65k | tg 4k/16k/65k |
|---|---|---|
| mainline `--cuda-only` (f16 KV) | ~305 | 16.4 |
| **ik_llama** (q8_0 KV, 24 threads) | **366 / 385 / 362** | **19.8 / 19.5 / 17.7** |

**+26% prefill and +19% generation** — enough to matter for the RAM-offload
case, which is the one configuration where this repo's engine is weakest.

Part of that gap is a capability difference rather than a speed difference:
ik runs the KV cache at q8_0, and **mainline segfaults with `-ctk q8_0` on
this model past ~4k context** (reproduced twice; 4k alone gives 335 pp /
18.9 tg, 16k kills the server). Notably one of the ik commits from that same
week is titled *"Allow Q8_0 cache in the CUDA DSA implementation"*.

Three tuning variables were then isolated, one at a time, on the ik side:

| Variable | Result |
|---|---|
| **Threads 12 → 24** | pp 268.6 → 354.6 (**+32%**), tg flat at 24.6–25.2 |
| CUDA 13.3 → 12.8 (arch and threads fixed) | 382.0 → 384.6 pp, 19.71 → 19.47 tg — noise |
| `120-real` → `120a-real` | 376.2 → 384.6 pp — noise |

So the **Blackwell CUDA 13.x collapse documented in
[cuda-fa-blackwell.md](cuda-fa-blackwell.md) does not affect ik_llama** — its
MLA path does not go through the flash-attention kernels that misbehave on
sm_120. Only mainline needs the CUDA 12.8 container build.

Threads were the whole story, and only for prefill: generation is
bandwidth-bound and flat from 12 threads upward. The toolkit's own tuning
guide had concluded 18 threads was optimal because its table only measured
generation.

Structural limitation that decides the architecture question: **ik_llama has
no `GGML_BACKEND_DL`**, one backend per build, so it cannot host AMD and
NVIDIA in a single process. It is complementary to this repo rather than a
replacement — mainline for dual-vendor, ik for NVIDIA+RAM MoE — and it is the
only way to run the `-R4` repacked quants (ggml type 213) that mainline
rejects.
