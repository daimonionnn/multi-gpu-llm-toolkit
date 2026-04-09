import ctypes, os

# First load NVIDIA CUDA to simulate what llama-server does
try:
    nvcuda = ctypes.CDLL('nvcuda.dll')
    res = ctypes.c_int(0)
    nvcuda.cuInit(0)
    
    # Create a CUDA context on device 0 (NVIDIA)
    ctx = ctypes.c_void_p(0)
    dev = ctypes.c_int(0)
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    
    free_mem = ctypes.c_size_t(0)
    total_mem = ctypes.c_size_t(0)
    nvcuda.cuMemGetInfo_v2(ctypes.byref(free_mem), ctypes.byref(total_mem))
    print(f'NVIDIA CUDA initialized: {free_mem.value/(1024**3):.1f} GiB free / {total_mem.value/(1024**3):.1f} GiB total')
except Exception as e:
    print(f'CUDA init: {e}')

# Now test HIP allocations
hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

free_mem = ctypes.c_size_t(0)
total_mem = ctypes.c_size_t(0)
hip.hipMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
print(f'HIP: {free_mem.value/(1024**3):.2f} GiB free / {total_mem.value/(1024**3):.2f} GiB total')

for size_gb in [60, 65, 70, 72, 74, 76, 78, 80, 82, 84]:
    size = size_gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if err == 0:
        print(f'  hipMalloc({size_gb} GiB): SUCCESS')
        hip.hipFree(ptr)
    else:
        print(f'  hipMalloc({size_gb} GiB): FAILED (err={err})')
        break
