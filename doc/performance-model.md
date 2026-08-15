# What actually limits prefill and generation

Every number in [benchmarks.md](benchmarks.md) is an instance of a small number
of mechanisms. This file is the mechanisms — what prefill and generation are
doing, which piece of hardware each one leans on, and what the flags in this
toolkit change. Everything is illustrated with measurements from the two rigs in
[systems.md](systems.md), because the same model on the same GPU behaves
completely differently on them, and the reasons are instructive.

Read this before tuning a new rig. It will save you most of the search.

## The two phases

A request has two phases with **opposite** bottlenecks. Confusing them is the
single most common way to tune the wrong thing.

| | Prefill (prompt processing, `pp`) | Generation (decode, `tg`) |
|---|---|---|
| Work | All prompt tokens at once | One token at a time |
| Per weight loaded | Used for hundreds/thousands of tokens | Used for exactly one token |
| Limited by | Compute, or getting weights to the compute unit | **Memory bandwidth**, almost always |
| Helped by | Bigger batch, faster link, faster GPU | Faster memory, fewer bytes per token |
| Felt as | The wait before the first token | The speed text appears at |

The formal version: prefill has high **arithmetic intensity** (many FLOPs per
byte of weight read), so it can saturate a GPU's compute. Generation at batch
size 1 has an arithmetic intensity of about 2 FLOPs per byte — hopeless — so the
GPU idles and the memory bus decides everything. This is why a card that pulls
400 W during prefill sits at 120 W during generation.

Consequence worth internalising: **generation speed is set almost entirely by
where the weights live, not by how fast the chip that reads them is.** A weak
GPU reading from fast memory beats a strong GPU reading from slow memory.

## Why MoE changes the picture

Dense models read every weight for every token. Mixture-of-Experts models — the
ones this project mostly runs — do not, and the asymmetry is large:

- **Generation** activates a handful of experts per token (DeepSeek V4 Flash:
  256 experts per layer, a few active). Only those bytes are read. A 146 GB
  model can generate as if it were a much smaller one.
- **Prefill** processes thousands of tokens at once, and between them they route
  to *every* expert. The whole expert tensor gets touched.

So for an MoE model, expert weights that live off the compute device are cheap
during generation and brutally expensive during prefill. Every layout decision
in this repo follows from that one sentence.

A second consequence: MoE models tolerate slow memory far better than dense ones
during generation. `gpt-oss-120b` (60 GB, 4 of 128 experts active) generates at
258 t/s on `dual-linux`, faster than a 27B dense model, because it *reads* very
little per token.

## The five candidate bottlenecks

| Resource | Sets prefill? | Sets generation? | How you recognise it |
|---|---|---|---|
| **GPU compute** | Yes, if all weights are already on it | Almost never | GPU at 90%+, high power draw |
| **Link between devices** (PCIe/TB) | Yes, if weights or activations cross it | Rarely (tiny transfers) | Both devices <50% busy, low power |
| **Memory bandwidth** where the weights are | Sometimes | **Yes, nearly always** | tg scales with bandwidth, flat vs context |
| **CPU compute** | Only with `--no-op-offload` | Only for CPU-hosted experts | CPU at 70%+, GPU idle |
| **Placement pathologies** | Yes | Yes | Numbers far below what all of the above predict |

That last row is not a hardware resource, and it is where most of the surprises
in this project came from: memory that is technically present but reached
through a staging buffer, fused kernels silently disabled, an allocation that
fails at a size the machine clearly has. See [Traps](#traps-that-look-like-hardware-limits).

## Arithmetic you can do before buying anything

### Generation

```
tg  ≈  1 / Σ_over_devices ( bytes_read_on_device / bandwidth_of_that_device )
```

Bytes read per token = active expert weights + attention/shared weights + the KV
cache slice. For an MoE with `E` experts of which `A` are active:

```
bytes_per_token ≈ (A/E) × expert_bytes + dense_bytes + kv_bytes
```

The practical form of this is a ladder, and `dual-linux` measured it directly on
DeepSeek V4 Flash — generation tracks **the number of expert layers that are not
in VRAM**, and almost nothing else:

| RAM-hosted expert layers | Quant | tg |
|---:|---|---:|
| 0 | IQ2_XXS, 80.8 GB | **74.5** |
| 8 | IQ3_XXS, 97 GB | 29.3 |
| 10 | IQ3_S, 108 GB | 27.4–28.4 |
| 18 | MXFP4, 146 GB | 16.4 |

Note how brutal the first step is: 0 → 8 layers costs 60%, far more than the
bandwidth ratio alone predicts, because each off-GPU layer adds synchronisation
and scheduling cost on top of the slower read.

The same arithmetic run backwards is useful. On this project's two rigs, the
identical CUDA-only layout (18 RAM-hosted layers) gives **16.4 tg** on
`dual-linux` (RAM measured at 89.9 GB/s) and **22.9 tg** on `halo-win`. If the
layout is purely bandwidth-bound, `halo-win`'s effective expert-read bandwidth
is `89.9 × 22.9/16.4 ≈ 125 GB/s`. That is well below LPDDR5X's nominal figure,
which suggests scattered expert reads do not stream at peak — and `halo-win`'s
RAM bandwidth has never been measured directly (there is no Windows port of
`linux/scripts/membw.c`). Worth doing.

### Prefill when the weights are not on the compute device

This is the case that dominates large-MoE rigs. With `--op-offload` (default
**on**), llama.cpp does not compute CPU-hosted expert layers on the CPU — it
ships the *weights* to the GPU and computes there. So:

```
pp_max  ≈  batch_tokens × link_bandwidth / offloaded_weight_bytes
```

Worked for DeepSeek V4 Flash MXFP4, `--n-cpu-moe 18` (a 58.4 GiB host buffer —
you can read the exact figure out of a failed load), `-b 4096`:

| Rig | Link | Bandwidth | Predicted ceiling | Measured |
|---|---|---:|---:|---:|
| `halo-win` | PCIe 4.0 x4 | ~7 GB/s | ~460 pp | **299** |
| `dual-linux` | PCIe 5.0 x16 | ~50 GB/s | ~3300 pp | **1175** |

The model is an upper bound, not a predictor — real efficiency landed at 65% and
36% of it, because transfers overlap with compute and not every expert is
touched in every batch. What it *does* get right is the ranking and the order of
magnitude, and that is enough to decide a layout before buying hardware.

### Prefill when everything is already on GPUs

Then the link carries only activations, which are tiny by comparison: a 2048-token
micro-batch of a 7168-wide model is ~29 MB per crossing, about 1 GB per
micro-batch across 18 layers — a seventh of a second even on four lanes. Prefill
becomes GPU compute, split between the cards, and the slow card sets the pace.

## Case studies: what each resource actually bought

All rows: DeepSeek V4 Flash MXFP4 (146 GB), `-c 131072`, `-b 4096 -ub 2048`,
pp/tg at 16k context.

### 1. Link width — the biggest single factor for prefill

| Rig | Link | Layout | pp | tg |
|---|---|---|---:|---:|
| `dual-linux` | PCIe 5.0 x16 | CUDA-only, `-ncmoe 18` | **1175** | 16.4 |
| `halo-win` | PCIe 4.0 x4 | CUDA-only, `-ncmoe 18` | 299 | **22.9** |

Same GPU, same model, same flags. **3.9x the prefill for 8x the link**, and the
generation goes the other way because that rig's RAM is slower. This is the
cleanest demonstration in the project that prefill in a weight-streaming layout
is a bandwidth problem, not a compute problem.

If you are specifying a machine for large MoE models: **lanes to the GPU matter
more than the GPU**, as long as any weights live off it.

### 2. Same width, different cable — Thunderbolt vs OCuLink

| Connection | pp | tg |
|---|---:|---:|
| Thunderbolt 5 tunnel | 260.7 | 22.70 |
| OCuLink, PCIe 4.0 x4 | 299.1 | 22.93 |

Both present as `gen4 x4`. Going native bought **+15% prefill and ~1%
generation** — the tunnel costs latency and protocol overhead, and that is all.
An earlier revision of these docs blamed Thunderbolt for the gap against
`dual-linux`; this measurement is what disproved it. **The lane count is the
constraint, not the connector.**

(Watch out for one artefact: `nvidia-smi` reports `pcie.link.gen.current 1` when
the GPU is idle, because the link downclocks. Sample it during a prefill.)

### 3. RAM bandwidth — the generation lever

Two independent measurements:

- **Across rigs.** The CUDA-only layout generates 16.4 t/s on `dual-linux`
  (DDR5, 89.9 GB/s) and 22.9 t/s on `halo-win` (LPDDR5X): **+40%** from memory
  alone, on identical GPU and flags.
- **Within one rig.** `dual-linux` went from 4 mixed DIMMs at 6267 MT/s to 2
  matched at 7400: measured bandwidth 74.6 → 89.9 GB/s (**+20.5%**), with a
  capacity cost that stopped the 146 GB model fitting the page cache.

RAM bandwidth does **not** help prefill in the `--op-offload` layout, because the
CPU never computes there — the weights are read once and shipped. It is a pure
generation lever.

### 4. A second GPU — even a weak one, even sharing the same RAM

`halo-win`'s iGPU has no memory of its own; it reads the same LPDDR5X the CPU
does. It should be worth nothing. It is worth a great deal:

| Layout | Where the 18 overflow expert layers live | pp | tg |
|---|---|---:|---:|
| CUDA-only | System RAM, shipped over the link per batch | 299.1 | 22.93 |
| `vulkan-cuda` | iGPU | 352.1 | 34.35 |
| **`rocm-cuda`** | iGPU | **497.5** | **36.13** |

**+66% prefill, +58% generation** — with the same bytes in the same physical
memory. Nothing changed except *who computes on them*: instead of copying 58 GiB
across four lanes into the big GPU, a second GPU reads them in place.

Two lessons generalise:

- A second GPU is worth having even if it is slow and even if its memory is your
  system RAM, provided the model does not fit the first one.
- The two backends are **not** interchangeable. HIP beat Vulkan by 41% on prefill
  on identical placement. If a rig can run ROCm, measure it.

The converse also holds, and `dual-linux` measured it: when a model *does* fit
one card, adding the second one costs 3x (gpt-oss-120b, 258 tg alone against
~85 tg in dual). **Only split what does not fit.**

### 5. CPU and AVX-512 — almost irrelevant, and measurably so

This one surprises people with strong CPUs. `halo-win` has 16 Zen 5 cores with
AVX-512 and LPDDR5X; `dual-linux` has an Arrow Lake without AVX-512 and slower
DDR5. It does not matter, because with `--op-offload` the CPU does not compute
the experts — it only holds them.

Measured on `dual-linux` (interleaved, ~330k prefill tokens), 24 threads vs 4:

| Context | pp change | tg change |
|---:|---:|---:|
| 4 096 | +0.5% | +4.5% |
| 16 384 | +0.2% | +3.9% |
| 32 768 | −0.1% | +2.0% |

**Threads are worth nothing on prefill and ~3% on generation.**

And if you force the CPU to actually do the work with `--no-op-offload`, on the
machine with the better CPU:

| `halo-win`, CUDA-only | pp 4k | pp 16k | tg 4k | tg 16k |
|---|---:|---:|---:|---:|
| `--op-offload` (default) | 266.4 | 299.1 | 23.29 | 22.93 |
| `--no-op-offload`, 16 threads | 99.4 | 100.1 | 23.85 | 23.68 |
| `--no-op-offload`, 32 threads | 98.4 | 99.8 | 23.64 | 23.44 |

**Three times slower**, and SMT adds nothing. Copying 58 GiB of weights across
four PCIe lanes and computing on a Blackwell is three times faster than
computing in place on a modern AVX-512 CPU. Generation is 3% *better* without
op-offload, which fits: at batch 1 there is nothing to amortise, so the transfer
is pure overhead.

The ranking on this rig, then, is not about the link at all — it is about **which
processor does the expert matmuls**:

1. A second GPU that owns the memory — 497 pp
2. The big GPU, weights shipped to it — 299 pp
3. The CPU — 100 pp

### 6. Where the second GPU's memory lives

Same iGPU, same layout, only the BIOS carve-out changed:

| Framebuffer | `vulkan-cuda` pp | `rocm-cuda` pp |
|---|---:|---:|
| 1 GB (GTT, shared) | **352.1** | **497.5** |
| 64 GB (dedicated carve-out) | 188.0 | 468.1 |

A dedicated carve-out that is not fully CPU-visible (`isLargeBar: 0`) makes every
host write into it go through a small BAR window and a staging buffer. Vulkan
pays 45% of its prefill for that; HIP pays ~6%. Utilisation confirms the
mechanism — at 64 GB *both* devices got less busy (NVIDIA 33%→28%, iGPU 48%→34%)
and drew less power. They were waiting on copies.

Generalisation: on a UMA machine, **giving the iGPU "more VRAM" can make it
slower**. What matters is not how much memory it is allotted but whether writes
into that memory need staging.

### 7. GPU software can be worth more than GPU hardware

The CUDA toolkit used to *build* llama.cpp changed token generation by a factor
of two on Blackwell — same source, same driver, same card
([cuda-fa-blackwell.md](cuda-fa-blackwell.md)):

| Depth | CUDA 13.3 build | CUDA 12.8 build |
|---:|---:|---:|
| 7168 | 71.9 | 74.9 |
| 8192 | **34.7** | **74.6** |
| 32768 | ~13 | 69.6 |

Worth knowing that this is **model-shape-specific**: it lives in a flash-attention
heuristic for GQA models, and DeepSeek V4 Flash's MLA path never touches it —
measured on `halo-win`, CUDA 12.4 and 13.3 agree within noise and there is no
cliff at all. Never generalise a kernel-level finding across model architectures
without re-measuring.

### 8. Model shape beats model size

From `dual-linux`, all on the same card: a 104 GB model (Step-3.7-Flash, 8 of 288
experts active) generates at 78–94 t/s while an 80 GB one (DeepSeek 2-bit, more
active weight per token) manages 74.5. **What you read per token decides, not
what you store.** When choosing a model for a bandwidth-limited rig, look at
active parameters, not file size.

## Traps that look like hardware limits

Four failures in this project produced numbers or crashes that no amount of
hardware reasoning would have explained:

1. **Fused kernels silently disabled.** Putting whole layers on the iGPU
   (`-ts` without `-ot`) moved attention there too; the Vulkan backend implements
   none of DeepSeek V4's four fused ops, so llama.cpp disabled all of them
   globally and the run died in warmup. Fix: move *only* `ffn_*_exps` tensors.
   The log says so explicitly (`resolve_fused_ops: ... set to disabled`) — read it.
2. **Pinned memory has to be decided per layout.** Dual layouts need pinning on,
   or the iGPU's single large allocation fails or hangs; CPU-offload layouts need
   it off, because pinning one 58.4 GiB host buffer fails outright with 112 GB
   free. Large single allocations are fragile on Windows either way — the same
   unpinned load failed once and succeeded on retry.
3. **A runtime that lists no devices** is usually a `PATH` problem (HIP resolving
   rocBLAS out of the SDK), not the ABI mismatch it looks like.
4. **Idle telemetry lies.** `pcie.link.gen.current` reads 1 on an idle GPU.

## Parameter reference

### Placement — where each tensor lives

| Flag | What it does | When it matters |
|---|---|---|
| `-ngl N` / `--n-gpu-layers` | How many layers go to GPU. `99` = all | Always set 99 and control placement with the flags below |
| `--device D1,D2` | Which devices, in which order. `-ts` and `-ot` refer to this order | Multi-GPU, or picking one of several Vulkan devices |
| `-ncmoe N` / `--n-cpu-moe` | Keep expert weights of the first N layers in system RAM | The simplest way to fit an oversized MoE; costs prefill via the link |
| `-ot RE=BUF` / `--override-tensor` | Send tensors matching a regex to a named buffer (`ROCm0`, `Vulkan0`, `CUDA0`, `CPU`) | Precise control. `blk\.(2[5-9]\|3[0-9]\|4[0-2])\.ffn_.*_exps.*=ROCm0` puts the experts of layers 25–42 on the AMD device and nothing else |
| `-ts A,B` / `--tensor-split` | Proportional layer split across devices | Use `-ts 0,1` with `-ot` to mean "everything on the second device except what `-ot` overrides". A bare `-ts` split is what breaks DeepSeek V4 |

### Batching — the prefill lever

| Flag | What it does | When it matters |
|---|---|---|
| `-b N` / `--batch-size` | Logical batch: tokens submitted per step | Bigger = weights amortised over more tokens |
| `-ub N` / `--ubatch-size` | Micro-batch actually computed at once | **The** prefill lever when weights stream over a link; irrelevant when they do not |
| `-np N` / `--parallel` | Server slots for concurrent requests | Throughput serving; splits the KV cache |

Measured both ways. On `dual-linux`, where experts stream over PCIe,
`-b 4096 -ub 2048` against llama.cpp's default `2048/512` was worth **+55–60% of
prefill for free**:

| `-b`/`-ub` | pp 4k | pp 16k | pp 32k | tg 4k |
|---|---:|---:|---:|---:|
| 2048 / 512 (default) | 565 | 595 | 597 | 25.5 |
| **4096 / 2048** | **937** | **947** | **907** | 24.6 |
| 8192 / 4096 | 964 | OOM at 16k | — | 24.3 |

On `halo-win` with everything GPU-resident, raising it further **lost** on both
axes, because nothing is being transferred to amortise and the bigger compute
buffers cost an expert layer of VRAM. Same flag, opposite sign, and the rule is:
`-ub` buys prefill exactly to the extent that weights cross a link.

### Attention and cache

| Flag | What it does | Notes |
|---|---|---|
| `-c N` / `--ctx-size` | Context window | KV is preallocated: a bigger window permanently costs VRAM you could have given to experts |
| `-fa on\|off\|auto` | Flash attention | Keep `on`. On Blackwell CUDA, `off` aborts the backend outright |
| `-ctk`/`-ctv TYPE` | Quantise the KV cache | Frees VRAM at some quality cost; MLA models like DeepSeek have tiny KV anyway (~13 GB at 256k) so rarely needed |
| `--cache-reuse N` | Reuse KV across requests via shifting | Helps agent workloads that resend a growing prefix |

The context/expert trade is real and measurable: at `-c 131072` this rig fits 18
expert layers on the iGPU, and 17 no longer loads. `dual-linux` found the same
shape — dropping from 256k to 128k freed enough VRAM for two more expert layers
and made the *smaller* context **faster** (480–592 pp vs 386).

### Loading

| Flag | What it does | Notes |
|---|---|---|
| `-lm MODE` / `--load-mode` | `auto` \| `none` \| `mmap` \| `mlock` | Replaces the deprecated `--no-mmap`/`--mlock`/`--direct-io` |
| | `none` = read into anonymous memory | Use when CPU-hosted tensors exceed the page cache, so mmap would page them back off NVMe |
| | `mmap` = map the file | Right when everything lands in VRAM: faster load, shared page cache |

### Compute

| Flag | What it does | Notes |
|---|---|---|
| `--op-offload` / `--no-op-offload` | Whether host-resident tensor ops are shipped to a device (default: on) | Leave on. Off is a diagnostic: it tells you how much of your prefill is transfer |
| `-t N` / `--threads` | Generation threads | Worth ~3% when experts are CPU-hosted |
| `-tb N` / `--threads-batch` | Prefill threads | Worth ~0% with op-offload on |
| `--jinja` | Use the model's own chat template | Needed for correct tool-calling and reasoning formats |

### Toolkit wrappers

| Script / setting | Purpose |
|---|---|
| `windows/scripts/start-llama-server.ps1` | Resolves devices per `-Mode` (`rocm`, `cuda`, `vulkan`, `rocm-cuda`, `vulkan-vulkan`, `vulkan-cuda`), sets backend env, prepends `$HIP_PATH\bin` for ROCm modes, passes `-ExtraArgs` through to `llama-server` |
| `-RuntimeDir` | Point at any runtime directory, including hand-assembled prebuilt mixes |
| `-AllowPinned` | Suppress `GGML_CUDA_NO_PINNED=1`. Needed by dual layouts, harmful to big CPU-offload ones |
| `windows/scripts/start-deepseek-mxfp4-nvidia-amd.ps1` | Model profile. `-Layout rocm-cuda\|vulkan-cuda\|cuda-only` picks placement and the matching runtime, and decides pinning |
| `windows/scripts/setup-llama.ps1 -Backend X` | Builds a backend into its own `runtime-X/`; several coexist and you switch at launch |
| `windows/scripts/benchmark-loaded-model.ps1` | HTTP benchmark against a running server — the numbers in this file |
| `windows/scripts/run-llama-bench.ps1` | `llama-bench`, for raw hardware capability without HTTP |
| `LLM_MODELS_DIR` | Root holding `<publisher>/<repo>/` GGUF folders; defaults to LM Studio's directory |

## Choosing a layout for a new rig

1. **Does the model fit one GPU's VRAM?** If yes, use one GPU. Adding a second
   costs up to 3x. Stop here.
2. **How much overflow is there?** That volume, times how often it must move,
   is your prefill budget.
3. **Is there a second GPU — any second GPU, including an iGPU?** Put the
   overflow experts there with `-ot`, not in CPU RAM. Measured worth +66%
   prefill and +58% generation here, and it does not matter that the iGPU reads
   the same system RAM.
4. **Only expert FFN tensors**, never whole layers, or you will lose fused
   attention kernels and possibly the run.
5. **If there is no second GPU**, use `-ncmoe` and then tune `-ub` upward until
   it OOMs at your target context — that is the amortisation lever.
6. **Fill the fast card first.** Each expert layer you can move back to it is
   worth roughly 4.5% prefill and 2% generation. The ceiling is the KV cache, so
   decide the context window deliberately.
7. **Measure both backends** if the AMD device supports both. HIP beat Vulkan by
   41% on prefill in the case measured here.
8. **Then check for the traps** above before believing a disappointing number.

## Rules of thumb

- Generation speed ≈ bandwidth of the memory holding your weights. Nothing else
  comes close.
- Prefill speed ≈ how much weight has to move, over how fast a link, unless
  nothing has to move — then it is GPU compute.
- The CPU is a warehouse, not a worker. Do not pay for cores to run MoE offload.
- Lanes to the GPU are worth more than GPU generation, once anything spills.
- Bigger context is not free; it is expert layers you no longer fit.
- Any number that is far off what the arithmetic here predicts is a bug, a
  driver behaviour, or a disabled kernel — go read the load log.
