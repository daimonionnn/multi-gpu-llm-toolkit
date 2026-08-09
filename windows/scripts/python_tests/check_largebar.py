import ctypes, os, struct

# Load patched HIP DLL
hip = ctypes.CDLL(r'amdhip64_7.dll')

hip.hipSetDevice(0)

# hipDeviceProp_t is a huge struct - allocate 4096 bytes
props = ctypes.create_string_buffer(4096)
err = hip.hipGetDeviceProperties(props, 0)
print(f'hipGetDeviceProperties: err={err}')

# Dump device name
name = props.raw[:256].split(b'\x00')[0].decode()
print(f'Device: {name}')

# Search for isLargeBar. It's an int (4 bytes).
# We know totalGlobalMem is ~110 GiB. Let's find known values to orient ourselves.
# totalGlobalMem is size_t at some known offset (typically around offset 256-280)
totalMem_bytes = 110456 * 1024 * 1024  # ~110456 MiB from device init log

# Scan for totalGlobalMem value to find its offset
for off in range(256, 1024, 8):
    val = struct.unpack_from('<Q', props.raw, off)[0]
    if abs(val - totalMem_bytes) < 1024*1024*1024:  # within 1 GiB
        print(f'  totalGlobalMem at offset {off}: {val} ({val/(1024**3):.2f} GiB)')

# isLargeBar should be near the end of the struct. 
# In HIP SDK 7.1, the field order from hip_runtime_api.h has isLargeBar fairly late.
# Let me dump all int-sized fields that are 0 or 1 (boolean-like) in a range
print('\nScanning for isLargeBar (should be 0 or 1):')
print('Looking for fields with value 0 or 1 in range [800, 2000]:')
prev_val = None
for off in range(800, 2000, 4):
    val = struct.unpack_from('<i', props.raw, off)[0]
    if val in (0, 1):
        print(f'  offset {off}: {val}')
