import ctypes

hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

# Test hipMallocManaged vs hipMalloc
print('=== hipMalloc ===')
for size_gb in [60, 70, 74, 76, 78, 80, 82]:
    size = size_gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p(0)
    err = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    status = 'OK' if err == 0 else f'FAIL(err={err})'
    print(f'  {size_gb} GiB: {status}')
    if err == 0:
        hip.hipFree(ptr)

print('\n=== hipMallocManaged ===')
# hipMallocManaged(void** ptr, size_t size, unsigned int flags)
# flags: hipMemAttachGlobal = 0x01
for size_gb in [60, 65, 70, 72, 74, 76, 78, 80, 82]:
    size = size_gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p(0)
    err = hip.hipMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(1))
    status = 'OK' if err == 0 else f'FAIL(err={err})'
    print(f'  {size_gb} GiB: {status}')
    if err == 0:
        hip.hipFree(ptr)
    else:
        break
