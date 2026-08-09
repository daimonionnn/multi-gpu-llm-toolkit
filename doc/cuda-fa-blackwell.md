# CUDA token generation collapses at 8192 context on Blackwell (sm_120)

On the **`dual-linux`** rig, llama.cpp's CUDA backend loses half to four fifths of
its token-generation throughput the moment the KV cache reaches 8192 tokens. The
Vulkan backend on the *same* GPU is unaffected.

Root cause identified, mechanism confirmed by measurement, and a one-line local
fix verified. Not found in upstream issues as of 2026-08-09.

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

## Cause

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

### Applying it

The patch is **not** applied by `setup-llama.sh`, so that `git pull --ff-only`
keeps working on a clean tree. To use it:

```bash
cd linux/llama.cpp
git apply ../patches/0001-cuda-fa-exempt-blackwell-from-ada-mma-heuristic.patch
cd .. && ./scripts/setup-llama.sh --backend rocm-cuda
```

A prebuilt patched runtime from this investigation is kept at
`linux/runtime-cuda-fapatch/` (gitignored) so the two can be compared directly.

## Scope

Affects any model with `gqa_ratio > 4` — which is most current models — running
on the CUDA backend on Blackwell, once context passes 8192. Prompt processing is
also slow on this path but does not have a cliff, since it never used the vector
kernel to begin with.

Not affected: Vulkan on the same card, ROCm and Vulkan on the AMD card, and any
context below 8192.

## Related upstream reports

Blackwell's FA/MMA paths already have known problems, though none of these is
this bug:

- [#24399](https://github.com/ggml-org/llama.cpp/issues/24399) — sm_120 out-of-range shared-memory store in the `mul_mat_q` MMA write-back epilogue
- [#21564](https://github.com/ggml-org/llama.cpp/issues/21564) — Xid 43 crash on RTX 5090 traced to `flash_attn_stream_k_fixup`
- [#24485](https://github.com/ggml-org/llama.cpp/issues/24485) — silent CPU fallback for quantized KV without `GGML_CUDA_FA_ALL_QUANTS`

A search of llama.cpp issues and discussions on 2026-08-09 found no existing
report of this performance cliff. It looks reportable upstream.

## Separate bug found alongside

`-fa off` aborts the CUDA backend outright at any depth, including 0:

```
CUDA error: invalid argument
  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2374
```

Reproduced on two models. In practice the CUDA backend is flash-attention-only
on this card, so the workaround of disabling FA is not available.
