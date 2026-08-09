# CUDA token generation collapses at 8192 context on Blackwell (sm_120)

> ### Status: cause narrowed to the CUDA toolkit, not the heuristic
>
> The original conclusion below - that llama.cpp's Ada-tuned flash-attention
> heuristic is at fault - **did not survive checking** and is retained only as a
> record of the reasoning. See [What it actually is](#what-it-actually-is).
>
> The measurements are all valid. The explanation was wrong.

## Symptom

Token generation, RTX PRO 6000 Blackwell, flash attention forced on:

| Depth | CUDA tg (t/s) | Vulkan tg (t/s) | CUDA / Vulkan |
|---|---:|---:|---:|
| 4096 | 74.4 | 71.2 | 104% |
| 6144 | 72.9 | 70.7 | 103% |
| 7168 | 72.0 | 70.3 | 102% |
| **8192** | **34.7** | 70.2 | **49%** |
| 10240 | 30.6 | 69.6 | 44% |
| 12288 | 27.4 | 68.9 | 40% |

*Qwen3.6-27B UD-Q4_K_XL. Vulkan is flat across the whole range, which rules out
any depth-related memory or hardware effect.*

The step is a cliff, not a slope, and it lands exactly on 8192.

It is not model-specific. Hermes-4-70B Q4_K_M (head dim 128, gqa 8) collapses
harder at the same boundary:

| Depth | CUDA tg | Vulkan tg | CUDA / Vulkan |
|---|---:|---:|---:|
| 7168 | 32.2 | 33.0 | 98% |
| **8192** | **9.4** | 33.7 | **28%** |
| 16384 | 5.5 | 31.1 | 18% |

## The GPU is idle, not busy

Sampling power during generation at depth 32768, same card:

| Backend | Avg power | Peak | tg |
|---|---:|---:|---:|
| CUDA | 210 W | 211 W | 13.2 t/s |
| Vulkan | 410 W | 503 W | 64.3 t/s |

The card is healthy — it pulls 392 W on prefill and 503 W under Vulkan, with no
throttling reported. Under CUDA at depth it simply is not being fed. Low power
draw is the most visible symptom of this bug.

## The hypothesis that failed

`ggml/src/ggml-cuda/fattn.cu`, in `ggml_cuda_get_best_fattn_kernel()`:

```c
if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1
    && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
    return BEST_FATTN_KERNEL_VEC;
}
...
return BEST_FATTN_KERNEL_MMA_F16;
```

For batch-size-1 decoding, the fast vector kernel is used — *unless* the model
has `gqa_ratio > 4` and the KV cache has reached 8192, in which case the tensor-core
MMA kernel is chosen instead. That is the exact boundary measured above.

When the carve-out was introduced ([13aeb7aef](https://github.com/ggml-org/llama.cpp/commit/13aeb7aef),
2025-08-20, *"CUDA: refactor FA support/selection code"*, [#15454](https://github.com/ggml-org/llama.cpp/pull/15454))
the condition was spelled out in a variable whose name says what it is:

```c
const bool mma_faster_for_rtx4000 = Q->ne[3] > 1 || (gqa_ratio > 4 && K->ne[1] >= 8192);
```

It is an **Ada-tuned heuristic**, applied to every `cc >= GGML_CUDA_CC_ADA_LOVELACE`.
The underlying "MMA beats vector for GQA at batch 1" result comes from
[#12014](https://github.com/ggml-org/llama.cpp/pull/12014), which was benchmarked
on **RTX 3090 and RTX 4090**.

`GGML_CUDA_CC_BLACKWELL` was not added to `common.cuh` until
[c8a2417d7](https://github.com/ggml-org/llama.cpp/commit/c8a2417d7) (2025-12-24) —
*after* the heuristic. Blackwell therefore inherited an Ada tuning that nobody
re-validated on it.

### Why MMA is the wrong choice on this chip

sm_120 (workstation/consumer Blackwell) is not sm_100 (datacenter Blackwell).
Per the [Blackwell GPU Wiki](https://0xsero.github.io/blackwell-gpu-wiki/kernels/flashattention/):

- sm_120 caps shared memory at 99 KiB, forcing **smaller FA tiles** than sm_100.
- FA-2 on sm_120 uses `mma.sync` only, with **no async tensor-core overlap**.
- The result is "lower per-step throughput", and it hurts most in
  **long-context decode at batch 1, where the workload is memory-bound on KV reads**.

That is precisely the case the heuristic switches *into* MMA for.

## Fix

Exempt Blackwell from the Ada carve-out —
[`linux/patches/0001-cuda-fa-exempt-blackwell-from-ada-mma-heuristic.patch`](../linux/patches/0001-cuda-fa-exempt-blackwell-from-ada-mma-heuristic.patch):

```c
const bool mma_faster = (gqa_ratio > 4 && K->ne[1] >= 8192) && cc < GGML_CUDA_CC_BLACKWELL;
if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !mma_faster) {
    return BEST_FATTN_KERNEL_VEC;
}
```

Measured after rebuilding only `fattn.cu`:

| Model | Depth | Stock | Patched | Speedup |
|---|---:|---:|---:|---:|
| Qwen3.6-27B UD-Q4_K_XL | 8192 | 34.7 | **71.1** | 2.05× |
| Qwen3.6-27B UD-Q4_K_XL | 12288 | 27.4 | **68.1** | 2.49× |
| Hermes-4-70B Q4_K_M | 8192 | 9.4 | **31.7** | 3.37× |
| Hermes-4-70B Q4_K_M | 16384 | 5.5 | **28.1** | 5.11× |

The cliff disappears completely; the curve becomes as flat as Vulkan's.

Re-measured afterwards across the whole configuration matrix (Qwen3.6-27B
UD-Q4_K_XL, tg128 at depth 32768, one run each, nothing else on the GPUs):

| Config | Stock | Patched | |
|---|---:|---:|---:|
| cuda (NVIDIA) | 13 | **56** | 4.24x |
| rocm-cuda (dual) | 14 | **46** | 3.33x |
| vulkan-cuda (dual) | 13 | **41** | 3.27x |
| vulkan (NVIDIA) | 65 | 65 | control |
| vulkan-vulkan (dual) | 44 | 44 | control |
| vulkan (AMD) | 27 | 27 | control |
| rocm (AMD) | 25 | 25 | control |

Every configuration containing CUDA recovers; the four that cannot be touched by
the patch stay within 1%, which is what makes the first three trustworthy. Depth
0 and 4096 are unchanged (78 -> 77, 75 -> 74), as expected from a change that
only takes effect at KV >= 8192, and prompt processing is unchanged everywhere
because it never used the vector kernel.

**The patch does not make CUDA the best choice at long context.** At 32k,
Vulkan on the same card still leads (65 vs 56 t/s); CUDA is ahead at short
context (77 vs 73 at depth 0) and the two cross over somewhere in between. Among
dual configurations the patch does restore `rocm-cuda` to the top at 32k
(46 vs 44 for `vulkan-vulkan`), reversing the stock ordering.

### Applying it

`setup-llama.sh` applies everything in `linux/patches/` when given `--patches`:

```bash
./scripts/setup-llama.sh --backend rocm-cuda --patches
```

Without the flag the build is stock upstream, so both behaviours stay available.
Updates are handled too: the script reverses its own patches before
`git pull --ff-only` and re-applies them afterwards, so a patched checkout does
not block updating. If upstream moves the code and a patch stops applying, the
script warns and continues with stock behaviour rather than failing the build.

A prebuilt patched runtime from the investigation is also kept at
`linux/runtime-cuda-fapatch/` (gitignored) for side-by-side comparison.

## Scope

Affects any model with `gqa_ratio > 4` — which is most current models — running
on the CUDA backend on Blackwell, once context passes 8192. Prompt processing is
also slow on this path but does not have a cliff, since it never used the vector
kernel to begin with.

Not affected: Vulkan on the same card, ROCm and Vulkan on the AMD card, and any
context below 8192.

## What it actually is

LM Studio ships its own Linux CUDA build of llama.cpp. On the same card, same
model and same measurement it does **not** collapse: 72.6 t/s at 12k context
where our build gives 27.6. Its version string is `fe2adf0`, 79 commits behind
ours, and no flash-attention commit sits between the two - the heuristic above
is byte-identical in both.

Ruled out, each by direct test:

| Candidate | Test | Result |
|---|---|---|
| llama.cpp revision | `git log fe2adf0..HEAD -- ggml/src/ggml-cuda/fattn*` | no FA commits |
| `sm_120` vs `sm_120a` | gencode flags in `build.ninja` | both already `compute_120a` |
| Single vs multi-arch | rebuilt with LM Studio's six-arch list | collapses identically |
| HIP co-compiled | CUDA-only build | collapses identically |
| llama-bench vs HTTP | both methods, both builds | agree |
| Worse codegen | `cuobjdump -res-usage` on the exact kernel | REG 192 vs 192, 181 vs 180 |
| Our build process | our Vulkan build vs **official** `b10331` Vulkan build | within 1-2% on both GPUs |

That last row matters most: this repo builds llama.cpp correctly. Our Vulkan
binary matches upstream's own release of the identical commit (2663 vs 2646
pp512 on NVIDIA, 1020 vs 1007 on AMD).

**The only remaining difference is the CUDA toolkit.** We build with CUDA 13.3;
LM Studio's build reports `built with GNU 12.3.0`, i.e. a CUDA 12.x toolchain.

This is circumstantial - strong elimination, no direct proof - and it cannot be
closed on this machine:

- CUDA 12.x cannot compile here at all, for the glibc 2.43 reason in
  [cuda-glibc-243.md](cuda-glibc-243.md). It is older than 13.1, which already
  fails.
- NVIDIA's `ubuntu2604` repository carries only 13.x.
- Swapping LM Studio's `libggml-cuda.so` into our runtime fails: the backend
  will not load across a 79-commit ABI gap.

### What to do about it

- **Use Vulkan for the NVIDIA card.** It is unaffected, costs ~3% at short
  context, and is faster than patched CUDA past ~8k anyway.
- **Or use a CUDA 12.x-built binary**, such as the one LM Studio ships.
- The patch in `linux/patches/` still helps *this* build by 3.3-4.2x and its
  measurements stand, but it treats a symptom of the toolchain rather than a
  bug in llama.cpp. Do not report it upstream.

## Related upstream reports

Blackwell's FA/MMA paths already have known problems, though none of these is
this bug:

- [#24399](https://github.com/ggml-org/llama.cpp/issues/24399) — sm_120 out-of-range shared-memory store in the `mul_mat_q` MMA write-back epilogue
- [#21564](https://github.com/ggml-org/llama.cpp/issues/21564) — Xid 43 crash on RTX 5090 traced to `flash_attn_stream_k_fixup`
- [#24485](https://github.com/ggml-org/llama.cpp/issues/24485) — silent CPU fallback for quantized KV without `GGML_CUDA_FA_ALL_QUANTS`

A search of llama.cpp issues and discussions on 2026-08-09 found no existing
report of this performance cliff.

It has not been reported. Worth knowing if anyone revisits that decision: the
fix here exempts all of Blackwell, which is blunt. sm_100 (datacenter Blackwell)
has more shared memory and async tensor-core overlap, so the MMA path may well
be the right choice there; a narrower condition would need hardware to verify.
llama.cpp also asks contributors to open an issue for discussion before sending
a patch, and forbids AI-written issue text, PR descriptions and review replies.

## Separate bug found alongside

`-fa off` aborts the CUDA backend outright at any depth, including 0:

```
CUDA error: invalid argument
  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2374
```

Reproduced on two models. In practice the CUDA backend is flash-attention-only
on this card, so the workaround of disabling FA is not available.
