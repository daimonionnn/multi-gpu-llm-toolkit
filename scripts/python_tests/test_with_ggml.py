import ctypes, os

os.add_dll_directory(r'..\..\runtime-rocm-cuda')
os.add_dll_directory(r'C:\Program Files\AMD\ROCm\7.1\bin')

# Load ggml-hip.dll (which loads amdhip64_7.dll, rocblas.dll, etc.)
print('Loading ggml-hip.dll...')
try:
    ggml_hip = ctypes.CDLL(r'..\..\runtime-rocm-cuda\ggml-hip.dll')
    print('  ggml-hip.dll loaded')
except Exception as e:
    print(f'  Failed: {e}')

# Now load HIP directly and test allocations
hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

free_mem = ctypes.c_size_t(0)
total_mem = ctypes.c_size_t(0)
hip.hipMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
print(f'\nHIP memory: {free_mem.value/(1024**3):.2f} GiB free / {total_mem.value/(1024**3):.2f} GiB total')

# Test allocations after ggml-hip.dll is loaded
for size_gb in [60, 70, 72, 74, 76, 78, 80]:
    size = size_gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if err == 0:
        print(f'  hipMalloc({size_gb} GiB): OK')
        hip.hipFree(ptr)
    else:
        print(f'  hipMalloc({size_gb} GiB): FAIL (err={err})')
        break
