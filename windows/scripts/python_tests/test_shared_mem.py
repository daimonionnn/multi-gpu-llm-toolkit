import ctypes, os

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin')
os.add_dll_directory(r'..\..\runtime-rocm-cuda')

# Init NVIDIA with 30 GiB allocated
nvcuda = ctypes.CDLL('nvcuda.dll')
nvcuda.cuInit(0)
dev = ctypes.c_int(0)
ctx = ctypes.c_void_p(0)
nvcuda.cuDeviceGet(ctypes.byref(dev), 0)
nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

# Allocate various amounts on NVIDIA and test HIP max
hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

for nv_gb in [0, 10, 20, 28, 30]:
    # Clean up previous NVIDIA allocation 
    # Actually we need to track allocations properly
    pass

# Test 1: No NVIDIA allocation
print('=== NVIDIA alloc: 0 GiB ===')
for size_gb in [70, 75, 80, 82, 84]:
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_gb * 1024**3))
    if err == 0:
        print(f'  hipMalloc({size_gb} GiB): OK')
        hip.hipFree(ptr)
    else:
        print(f'  hipMalloc({size_gb} GiB): FAIL')
        break

# Test 2: Allocate 28 GiB on NVIDIA
nv_ptr = ctypes.c_ulonglong(0)
nvcuda.cuMemAlloc_v2(ctypes.byref(nv_ptr), ctypes.c_size_t(28 * 1024**3))
print('\n=== NVIDIA alloc: 28 GiB ===')
for size_gb in [60, 65, 67, 68, 69, 70, 72, 75]:
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_gb * 1024**3))
    if err == 0:
        print(f'  hipMalloc({size_gb} GiB): OK')
        hip.hipFree(ptr)
    else:
        print(f'  hipMalloc({size_gb} GiB): FAIL')
        break

# Free NVIDIA memory
nvcuda.cuMemFree_v2(nv_ptr)

# Test 3: If NVIDIA only takes 20 GiB
nv_ptr2 = ctypes.c_ulonglong(0)
nvcuda.cuMemAlloc_v2(ctypes.byref(nv_ptr2), ctypes.c_size_t(20 * 1024**3))
print('\n=== NVIDIA alloc: 20 GiB ===')
for size_gb in [70, 72, 75, 78, 80]:
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_gb * 1024**3))
    if err == 0:
        print(f'  hipMalloc({size_gb} GiB): OK')
        hip.hipFree(ptr)
    else:
        print(f'  hipMalloc({size_gb} GiB): FAIL')
        break
nvcuda.cuMemFree_v2(nv_ptr2)
