import ctypes, struct

hip = ctypes.CDLL(r'amdhip64_7.dll')
hip.hipSetDevice(0)

# Use hipDeviceGetAttribute to query isLargeBar directly
# hipDeviceAttributeIsLargeBar enum value - need to find it
# From the header, it's near the end of the enum. Let me try common values.

# Alternative: use hipDeviceGetAttribute with the right attribute ID
# Let's try to find it by querying hipDeviceAttributeIsLargeBar
# The enum starts at hipDeviceAttributeUnused1 = 0 and goes up.
# isLargeBar is near the end. Let me scan the enum.

val = ctypes.c_int(0)

# Try to query some known attributes first to orient ourselves
# hipDeviceAttributeMaxThreadsPerBlock = 1 (CUDA compat)
for attr_id in range(1, 5):
    err = hip.hipDeviceGetAttribute(ctypes.byref(val), attr_id, 0)
    if err == 0:
        print(f'  attr {attr_id}: {val.value}')

# Now search for isLargeBar. On an APU, isLargeBar should be 0 (unpatched) or 1 (patched)
# The attribute enum is in order of declaration in hip_runtime_api.h
# isLargeBar is near the end. Let me try common offsets.
# HIP attribute IDs might not match CUDA ones. Let me look for the actual enum value.

# From HIP SDK headers, hipDeviceAttribute enum typically has isLargeBar around 280-320
# Let's scan a range and print attributes that are 0 or 1
print('\nScanning for isLargeBar attribute (value should be 0 or 1):')
for attr_id in range(270, 330):
    val.value = -1
    err = hip.hipDeviceGetAttribute(ctypes.byref(val), attr_id, 0)
    if err == 0 and val.value in (0, 1):
        print(f'  hipDeviceAttribute[{attr_id}] = {val.value}')

# Also let's try the exact hipDeviceAttributeIsLargeBar
# From ROCm 7.1 header, isLargeBar is attribute 284 (approximate)
# Let me try values around there
print('\nDetailed scan [275-310]:')
for attr_id in range(275, 310):
    val.value = -1
    err = hip.hipDeviceGetAttribute(ctypes.byref(val), attr_id, 0)
    status = "OK" if err == 0 else f"err={err}"
    print(f'  hipDeviceAttribute[{attr_id}] = {val.value} ({status})')
