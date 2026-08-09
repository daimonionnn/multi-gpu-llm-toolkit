import ctypes, os

os.add_dll_directory(r'..\..\runtime-rocm-cuda')
os.add_dll_directory(r'C:\Program Files\AMD\ROCm\7.1\bin')

# Step 1: Load NVIDIA CUDA backend (ggml-cuda.dll) AND initialize it
print('=== Loading ggml-cuda.dll (NVIDIA) ===')
ggml_cuda = ctypes.CDLL(r'..\..\runtime-rocm-cuda\ggml-cuda.dll')
print('ggml-cuda.dll loaded')

# Step 2: Load HIP backend (ggml-hip.dll) AND initialize it
print('\n=== Loading ggml-hip.dll (ROCm) ===')
ggml_hip = ctypes.CDLL(r'..\..\runtime-rocm-cuda\ggml-hip.dll')
print('ggml-hip.dll loaded')

# Step 3: Now test HIP allocations
hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

free_mem = ctypes.c_size_t(0)
total_mem = ctypes.c_size_t(0)
hip.hipMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
print(f'\nHIP memory: {free_mem.value/(1024**3):.2f} GiB free / {total_mem.value/(1024**3):.2f} GiB total')

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

# Step 4: Also try calling the ggml_cuda_init exported by ggml-hip.dll
# This is what llama-server does when it loads the ROCm backend
print('\n=== Calling ggml-hip init function ===')
try:
    # The exported symbol might be ggml_backend_cuda_reg or ggml_backend_reg_init
    # Let's try to find the init function
    init_fn = getattr(ggml_hip, 'ggml_backend_reg_init', None)
    if init_fn is None:
        init_fn = getattr(ggml_hip, 'ggml_backend_cuda_reg', None)
    if init_fn:
        print(f'Found init function, calling...')
        # Don't actually call it as we don't know the signature
    else:
        print('No init function found by name')
except Exception as e:
    print(f'Error: {e}')

# Test again after potential init
hip.hipMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
print(f'\nHIP memory after init: {free_mem.value/(1024**3):.2f} GiB free / {total_mem.value/(1024**3):.2f} GiB total')
