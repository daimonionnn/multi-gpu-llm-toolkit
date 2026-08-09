import ctypes, os

# Load from System32 (patched) - this is what ggml-hip.dll will load
hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

count = ctypes.c_int(0)
hip.hipGetDeviceCount(ctypes.byref(count))
print(f'Devices: {count.value}')

# Test allocations
for size_gb in [40, 60, 70, 72, 75, 78, 80, 81, 82]:
    size = size_gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if err == 0:
        print(f'  hipMalloc({size_gb} GiB): SUCCESS')
        hip.hipFree(ptr)
    else:
        print(f'  hipMalloc({size_gb} GiB): FAILED (err={err})')
