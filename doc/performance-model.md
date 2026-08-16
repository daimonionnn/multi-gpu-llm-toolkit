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

Running that arithmetic backwards is where it gets interesting — and where it
caught an error in an earlier revision of this file. The identical CUDA-only
layout (18 RAM-hosted layers) gives **16.4–17.2 tg** on `dual-linux` and
**22.9 tg** on `halo-win`, which was attributed here to LPDDR5X being faster
than DDR5. Then both rigs were measured with the same benchmark
(`membw.c`, 16 threads, identical buffer and repeat count):

| Rig | Memory | Streaming read |
|---|---|---:|
| `dual-linux` | DDR5, 2×64 GB @ 7400 | **89.9 GB/s** |
| `halo-win` | LPDDR5X (Strix Halo) | **83.8 GB/s** |

By that benchmark the "faster" memory is slower. And a thread sweep accounted
for part of the rest: `dual-linux`'s figure was taken at 4 threads, and
`halo-win` at 4 threads gives 20.45 instead of 22.93. What is left is ~22% at
equal threads, with **no established cause** — candidates are access latency
rather than streaming bandwidth (expert reads are scattered, not sequential),
AVX-512 helping the MXFP4 unpack at batch 1, and different llama.cpp builds.

Two lessons worth more than the number itself:

- **A memory benchmark measures a kernel, not a machine.** `membw.c` is
  thread-limited well below the bus: on `halo-win` it reads 54 GB/s on 8
  threads, 83.8 on 16, 95.4 on 24 and 102.5 on 32 — still climbing when it runs
  out of logical CPUs, and far under LPDDR5X's nominal figure. Compare rigs at
  equal thread counts and treat the result as a lower bound.
- **Compare configurations, not adjectives.** "Faster RAM" predicted a 40%
  generation gain that measurement cut to 22%, most of the difference being a
  flag nobody had set.

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

The model is an upper bound, not a predictor — transfers overlap with compute
and not every expert is touched in every batch. What it *does* get right is the
ranking and the order of magnitude, which is enough to decide a layout before
buying hardware.

Run the same arithmetic backwards and it becomes a diagnostic: divide the
offloaded bytes by the measured time per batch and you get the throughput the
link actually delivered.

| Rig | pp @16k | s per batch | Implied throughput | Utilisation of… |
|---|---:|---:|---:|---|
| `halo-win` | 299 | 13.7 | 4.6 GB/s | 58% of gen4 x4 |
| `dual-linux` | 1175 | 3.5 | 18.0 GB/s | 57% of gen5 **x8** |
| | | | | 29% of gen5 x16 |

Two independent rigs landing on the same 57–58% utilisation is what a
transfer-bound layout looks like. The 29% reading would mean the link was mostly
idle — and then cutting to four lanes could not have cost 4x, which it did. So
this arithmetic says `dual-linux` was running its NVIDIA card at **x8**, not the
x16 recorded in [systems.md](systems.md): a board that splits CPU lanes when the
second GPU slot is populated. Useful property of a performance model — it can
tell you your hardware inventory is wrong.

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

### 3. RAM bandwidth — a generation lever, and easy to overstate

The clean measurement is within one rig, where nothing else moves. `dual-linux`
went from 4 mixed DIMMs at 6267 MT/s to 2 matched at 7400: measured bandwidth
74.6 → 89.9 GB/s (**+20.5%**), with a capacity cost (215 → 122 GiB) that stopped
the 146 GB model fitting the page cache.

**Number of DIMMs is a speed decision, not just a capacity one.** Four populated
slots force the memory controller to clock down; here dropping to two DIMMs
raised the achievable clock by 1000 MT/s above the kit's own rating and bought
20% of bandwidth. On any DDR5 platform, 2 fast DIMMs beat 4 slow ones for
generation unless you genuinely need the capacity.

The cross-rig comparison is the cautionary tale — see
[the arithmetic section](#generation): "LPDDR5X vs DDR5" looked like +40% and
survives as ~22% at equal threads, with the streaming benchmark actually
favouring the DDR5 rig. Attribute cross-machine differences only after
equalising the flags.

RAM bandwidth does **not** help prefill in the `--op-offload` layout, because the
CPU never computes there — the weights are read once and shipped. It is a
generation lever only.

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

**Threads are worth nothing on prefill and ~3% on generation.** `halo-win`
reproduces the shape at 4 vs 16 threads: prefill 277.9/305.4 against
266.4/299.1 (unchanged, 4 threads even marginally ahead), generation 20.45
against 22.93 — **11%**, larger than on the Intel rig but the same sign.

And if you force the CPU to actually do the work with `--no-op-offload`, on the
machine with the better CPU:

| `halo-win`, CUDA-only, 16 threads | pp 4k | pp 16k | tg 4k | tg 16k |
|---|---:|---:|---:|---:|
| `--op-offload` (default) | 266.4 | 299.1 | 23.29 | 22.93 |
| `--no-op-offload` | 99.4 | 100.1 | 23.85 | 23.68 |

**Three times slower.** Copying 58 GiB of weights across four PCIe lanes and
computing on a Blackwell is three times faster than computing in place on a
modern AVX-512 CPU. Generation is 3% *better* without op-offload, which fits: at
batch 1 there is nothing to amortise, so the transfer is pure overhead.

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

## Machines: what to expect from a given build

Two of these are measured; the rest are the model applied to hardware this
project does not have. **Estimates are marked as such** — they are meant for
deciding what to buy, not for quoting. All assume the same workload as above: a
146 GB MoE that does not fit one consumer-class card.

| Build | Model fits? | Prefill | Generation | Binding constraint |
|---|---|---|---|---|
| **Strix Halo + 96 GB GPU on x4** *(measured)* | No, ~58 GB spills | 497 | 36.1 | iGPU compute, once nothing crosses the link |
| **Intel + 96 GB GPU on x16 + 32 GB AMD** *(measured)* | No | 480–592 dual, 1175 CPU-offload | 21–25 dual, 16–17 CPU-offload | Link when offloading, AMD card when dual |
| 9950X + 96 GB GPU on x16, 2 DIMMs *(est.)* | No | ~1100–1200 | ~16–19 | Dual-channel DDR5 on generation |
| 9950X + 96 GB GPU on x16, 4 DIMMs *(est.)* | No | ~1100–1200 | ~14–17 | Memory clock forced down by DIMM count |
| Threadripper/EPYC + 96 GB GPU *(est.)* | No | ~1200+ | ~25–35 | Nothing, until the model grows |
| 2× 96 GB GPUs *(est.)* | **Yes** | thousands | 70+ | Nothing — the case worth aiming at |
| 2× 32 GB consumer GPUs on x8 *(est.)* | No, ~82 GB spills | ~600–900 | ~12–15 | Amount of spill |
| Strix Halo alone, no dGPU *(est.)* | Only ≤110 GB models | ~100–200 | ~10–20 | iGPU compute and LPDDR5X |

### Would a Ryzen 9950X help this rig?

This came up directly, with the reasoning: *the Intel rig has no AVX-512 and
prefills at 1175 while this one has AVX-512 and prefills at 299, so AVX-512 was
never the point.* That reasoning is **correct**, and the measurements above
support it twice over — threads are worth 0% on prefill, and forcing the CPU to
compute the experts is 3x *slower*. A faster CPU of the same core count buys
approximately nothing.

But the conclusion "so a 9950X would not help" misses which part of the platform
actually matters. A 9950X is not a CPU swap, it is a **platform swap**, and the
platform brings the one thing this rig lacks: **PCIe 5.0 x16 to the GPU**. In
the CPU-offload layout that is precisely the 4x prefill difference measured
between the two rigs.

What it would cost, though, is everything the Strix Halo brings:

| | halo-win today | 9950X build *(est.)* |
|---|---|---|
| Lanes to the GPU | 4 | 16 |
| Prefill, CPU-offload layout | 299 | ~1100–1200 |
| Usable second GPU | iGPU with 128 GB of reach | none worth using |
| Best layout available | experts on the iGPU | experts in RAM |
| **Prefill, best layout** | **497** | ~1100–1200 |
| **Generation, best layout** | **36.1** | ~16–19 |

So it is a trade, not an upgrade: **roughly 2–3x the prefill for roughly half the
generation**. Which side wins depends entirely on the workload — long prompts
(agents, RAG, code bases) feel prefill; conversation feels generation. On the
evidence in this repo, the machine that produced 36 t/s on a lossless 146 GB
model is the more pleasant one to talk to, and the one that prefills a 63k
conversation in ~2 minutes instead of ~35 seconds is the more painful one to
wait on.

The DIMM point in that reasoning is right and worth repeating: **four populated
slots cost memory clock**, and generation is a memory-bandwidth problem.
`dual-linux` measured exactly this — going from 4 mixed DIMMs to 2 matched ones
was +20.5% of bandwidth. A 9950X with 4 DIMMs would be the worst of both worlds
for generation.

If the goal is to fix prefill without giving up generation, the honest answers
are, in order: **more VRAM so nothing spills**, then **more lanes and more
memory channels** (Threadripper/EPYC/Xeon-W class), then a wider slot for the
existing card. A different desktop CPU is not on that list.

### Other builds, and what decides them

- **2× 96 GB GPUs.** The only build here where the 146 GB model fits entirely in
  VRAM. Everything in this document stops mattering: no link traffic, no CPU
  involvement, generation limited by VRAM bandwidth (~70+ t/s by analogy with
  the all-VRAM 2-bit quants measured at 74.5). If the budget exists, this is the
  answer, and it is the *only* configuration that makes the model cheap on both
  axes.
- **2× 32 GB consumer cards.** 64 GB of VRAM against a 146 GB model leaves more
  spill than `halo-win` has, on links that are usually x8 each. Expect prefill
  between the two measured rigs and generation below both. Two small cards do
  not substitute for one large one when the model is this size — what matters is
  total VRAM, not GPU count.
- **Threadripper / EPYC / Xeon-W.** The class that removes both constraints at
  once: x16 per GPU and 4–8 memory channels. Worth it only if the model must
  spill; if it fits in VRAM, the platform is irrelevant.
- **Strix Halo alone.** The iGPU can reach all 128 GB, so models up to ~110 GB
  run with no discrete card at all — but on iGPU compute, which is where the
  ~100–200 pp estimate comes from. This is the configuration to compare against
  when asking what the discrete card is worth on this machine: it is worth a
  great deal for prefill and less for generation.
- **AMD discrete + NVIDIA discrete** (`dual-linux`). Both cards have real VRAM
  and real bandwidth, so the expert-offload layout runs at full speed — 480–592
  pp / 21–25 tg there. Its problem is neither link nor memory but **stability**:
  every fault recorded in this project is on the AMD path under DeepSeek.

### The general shape

For a model that does not fit one GPU, ranked by what actually determines the
result:

1. **How much spills** — the single biggest factor for generation. Zero spill is
   worth 3–4x over heavy spill.
2. **Whether a second processor can compute on the spill in place** (a second
   GPU, even an iGPU sharing system RAM) — worth +66% prefill and +58%
   generation here.
3. **Link width**, if the spill has to move — 4x prefill between x4 and x16.
4. **Memory bandwidth and channel count**, for whatever still lives in RAM.
5. **CPU model, core count, AVX-512** — last, and much further down than it
   feels like it should be.

## The cheapest prefill is the one that does not happen

Everything above treats prefill as work to be made faster. The larger win is
usually to avoid it. llama.cpp's server keeps each slot's KV cache and reuses
the longest common **prefix** of the next request, which on `halo-win` measures:

| Request against a 24 000-token prompt | Processed | Time |
|---|---:|---:|
| Cold | 24 003 | 52.6 s |
| Same prompt, 1 400 tokens appended | 1 404 | **3.5 s** |
| Same prompt, first 10 tokens changed | 25 415 | 55.8 s |

The asymmetry is the whole point: **appended content is nearly free, changed
prefixes cost everything.** Two consequences worth designing around:

- Put stable material first — system prompt, tool definitions, retrieved
  documents — and let only the tail vary. An agent that rewrites its system
  prompt each turn pays full prefill every turn; the same content appended costs
  a fifteenth of that.
- Keep the number of live conversations at or below the slot count
  (`--parallel`, 4 by default). Returning to a conversation still held in a slot
  cost 0.11 s in the same test.

`--cache-reuse`, which reuses cache across a divergence via KV shifting, sounds
like it should rescue the third row. On this model it did not: tested at 256
against both a prepended prefix and a 400-token insertion mid-document, it
reused nothing. Measure it on your own model rather than assuming either way.

The same reasoning applies to `-c`. A larger context window is not free even
when unused: its KV cache is preallocated, and on a spilling model that VRAM is
expert layers you no longer fit.

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
