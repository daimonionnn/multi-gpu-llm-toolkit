# CUDA 13.1 cannot build against glibc 2.43 (Ubuntu 26.04)

Affects the **`dual-linux`** rig. Blocks every backend that uses CUDA —
`rocm-cuda` and `vulkan-cuda` — while leaving `rocm` and `vulkan` unaffected.

This is the Linux counterpart of the MSVC toolset problem documented for
`halo-win`: the GPU is new enough that only a recent toolkit targets it, but
that toolkit does not get along with the rest of the system.

## Symptom

Any CUDA compilation, down to an empty kernel, fails during the device pass:

```
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(206): error: exception
specification is incompatible with that of previous function "rsqrt"
(declared at line 629 of .../crt/math_functions.h)

/usr/include/x86_64-linux-gnu/bits/mathcalls.h(206): error: exception
specification is incompatible with that of previous function "rsqrtf"
```

CMake reports it as `Failed to detect a default CUDA architecture` or a broken
CUDA compiler, because even its compiler-identification probe fails.

## Cause

glibc 2.43 declares `rsqrt`/`rsqrtf` in `bits/mathcalls.h`:

```c
#if __GLIBC_USE (IEC_60559_FUNCS_EXT_C23)
...
__MATHCALL_VEC (rsqrt,, (_Mdouble_ __x));
#endif
```

CUDA 13.1's `crt/math_functions.h` declares the same names as device builtins,
without a matching exception specification, so the two collide.

The guard resolves to enabled through `__USE_GNU`:

```c
#if defined __USE_GNU || defined __STDC_WANT_IEC_60559_FUNCS_EXT__
# define __GLIBC_USE_IEC_60559_FUNCS_EXT 1
```

`g++` defines `_GNU_SOURCE` unconditionally for C++, so the declarations are
always visible. There is no supported way to switch them off.

## Why the obvious workarounds do not work

All tested on the rig, all rejected:

| Attempt                                                | Result |
|--------------------------------------------------------|-----|
| Older host compiler (`-ccbin g++-13`)                  | Fails identically — the conflict is in glibc headers, not gcc |
| `-Xcudafe --diag_suppress=incompatible_exception_spec` | Not suppressible; the diagnostic is emitted as a hard error |
| `-Xcudafe --diag_suppress=1444`                        | Same |
| `-D__STDC_WANT_IEC_60559_FUNCS_EXT__=1`                | Feature-test macros only enable; there is no "off" value |
| `-Xcompiler -U_GNU_SOURCE`                             | Compiles a bare kernel, then breaks libstdc++ — see below |

The `-U_GNU_SOURCE` route looks promising until real C++ is involved:

```
/usr/include/c++/15/cwchar(150): error: the global scope has no "fwide"
/usr/include/c++/15/cwchar(151): error: the global scope has no "fwprintf"
```

libstdc++ needs `_GNU_SOURCE`, so removing it trades one breakage for a worse
one. Do not ship this flag.

## Actual fix

Install a CUDA Toolkit new enough for glibc 2.43 — 13.2 or later — from
NVIDIA's own apt repository. Ubuntu 26.04 ships only 13.1 in `multiverse`, and
the machine has no NVIDIA repo configured, so `apt` cannot currently offer a
newer one.

`setup-llama.sh` already scans for the newest toolkit that supports the GPU's
compute capability, so once a newer one is installed under `/usr/local/cuda-*`
it will be picked up with no change to the scripts.

## Meanwhile

Use Vulkan for the NVIDIA card. `vulkan-vulkan` drives both GPUs through one
backend and is unaffected — the same fallback role Vulkan plays on `halo-win`
when ROCm misbehaves, arrived at from the opposite direction.

| Backend         | Buildable on `dual-linux` today |
|-----------------|---------------------------------|
| `rocm`          | Yes                             |
| `vulkan`        | Yes                             |
| `vulkan-vulkan` | Yes (uses the `vulkan` runtime) |
| `rocm-cuda`     | No — blocked by this bug        |
| `vulkan-cuda`   | No — blocked by this bug        |
