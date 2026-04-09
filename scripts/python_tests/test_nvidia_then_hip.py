import ctypes, os

# Simulate llama-server's dual-backend loading
# First load NVIDIA CUDA backend (which initializes the NVIDIA driver)
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin')
os.add_dll_directory(r'..\..\runtime-rocm-cuda')

try:
    print('Loading nvcuda.dll...')
    nvcuda = ctypes.CDLL('nvcuda.dll')
    nvcuda.cuInit(0)
    
    # Create CUDA context (this is what ggml_cuda_init does)
    dev = ctypes.c_int(0)
    ctx = ctypes.c_void_p(0)
    nvcuda.cuDeviceGet(ctypes.byref(dev), 0)
    nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    
    nv_free = ctypes.c_size_t(0)
    nv_total = ctypes.c_size_t(0)
    nvcuda.cuMemGetInfo_v2(ctypes.byref(nv_free), ctypes.byref(nv_total))
    print(f'  NVIDIA: {nv_free.value/(1024**3):.1f} GiB free / {nv_total.value/(1024**3):.1f} GiB total')
    
    # NOW allocate a typical NVIDIA buffer (~30 GiB) to simulate the model loading
    nv_alloc_size = 30 * 1024 * 1024 * 1024
    nv_ptr = ctypes.c_ulonglong(0)
    err = nvcuda.cuMemAlloc_v2(ctypes.byref(nv_ptr), ctypes.c_size_t(nv_alloc_size))
    print(f'  Allocated 30 GiB on NVIDIA: err={err}')
    
except Exception as e:
    print(f'  NVIDIA init failed: {e}')

# Then test HIP allocations
hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

free_mem = ctypes.c_size_t(0)
total_mem = ctypes.c_size_t(0)
hip.hipMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
print(f'\nHIP memory: {free_mem.value/(1024**3):.2f} GiB free / {total_mem.value/(1024**3):.2f} GiB total')

for size_gb in [60, 65, 70, 72, 74, 76, 78, 80]:
    size = size_gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if err == 0:
        print(f'  hipMalloc({size_gb} GiB): OK')
        hip.hipFree(ptr)
    else:
        print(f'  hipMalloc({size_gb} GiB): FAIL (err={err})')
        break
