# CUDA 13.1 cannot build against glibc 2.43 (Ubuntu 26.04)

> **Resolved on `dual-linux`.** CUDA 13.3 from NVIDIA's apt repository compiles
> cleanly against glibc 2.43, and all four backends now build. Kept because the
> distro-packaged toolkit still has this defect, so anyone starting from a
> stock Ubuntu 26.04 will hit it — and the rejected workarounds below are worth
> not repeating.

Affected the **`dual-linux`** rig with CUDA 13.1: every backend that uses CUDA
(`rocm-cuda`, `vulkan-cuda`) failed to build, while `rocm` and `vulkan` were
unaffected.

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

Install a CUDA Toolkit new enough for glibc 2.43 from NVIDIA's own apt
repository. Ubuntu 26.04 ships only 13.1 in `multiverse`; NVIDIA's repo for the
same release carries 13.3.

```bash
curl -sSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3
```

**Install `cuda-toolkit-13-3`, not `cuda` or `cuda-13-3`.** The latter two pull
in the NVIDIA display driver and would replace a working one. The
`cuda-toolkit-*` package is driver-free — verify before committing to it:

```bash
sudo apt-get install --dry-run cuda-toolkit-13-3 | grep -iE '^Inst (nvidia|libnvidia|cuda-drivers)'
```

Empty output means no driver is touched.

`setup-llama.sh` scans for the newest toolkit that supports the GPU's compute
capability, so the new one under `/usr/local/cuda-13.3` is picked up with no
change to the scripts and no `PATH` edits.

## Outcome

With CUDA 13.3 installed, all four backends build on `dual-linux`:

| Backend       | CUDA 13.1       | CUDA 13.3 |
|---------------|-----------------|-----------|
| `rocm`        | Built (no CUDA) | Builds    |
| `vulkan`      | Built (no CUDA) | Builds    |
| `rocm-cuda`   | Blocked         | Builds    |
| `vulkan-cuda` | Blocked         | Builds    |

`rocm-cuda` enumerates both vendors in a single process:

```
CUDA0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
ROCm0: AMD Radeon AI PRO R9700 (32624 MiB, 32550 MiB free)
```

If you are stuck on the distro toolkit, `vulkan-vulkan` remains a working
dual-GPU fallback — one backend driving both cards, the same role Vulkan plays
on `halo-win` when ROCm misbehaves, arrived at from the opposite direction.
