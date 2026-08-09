import ctypes, time

hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

# Test hipMallocManaged at 72 GiB - the exact size llama-server needs for 2.3:1 split
size_gb = 72
size = size_gb * 1024 * 1024 * 1024
ptr = ctypes.c_void_p(0)
print(f'hipMallocManaged({size_gb} GiB)...')
t0 = time.time()
err = hip.hipMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(1))
t1 = time.time()
if err == 0:
    print(f'  SUCCESS in {t1-t0:.1f}s at {ptr.value:#x}')
    hip.hipFree(ptr)
else:
    print(f'  FAILED: err={err} in {t1-t0:.1f}s')
    # Also test a smaller managed allocation
    for small_gb in [60, 50, 40, 30]:
        size2 = small_gb * 1024 * 1024 * 1024
        ptr2 = ctypes.c_void_p(0)
        print(f'hipMallocManaged({small_gb} GiB)...')
        t0 = time.time()
        err2 = hip.hipMallocManaged(ctypes.byref(ptr2), ctypes.c_size_t(size2), ctypes.c_uint(1))
        t1 = time.time()
        if err2 == 0:
            print(f'  SUCCESS in {t1-t0:.1f}s')
            hip.hipFree(ptr2)
            break
        else:
            print(f'  FAILED: err={err2} in {t1-t0:.1f}s')
