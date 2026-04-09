import struct

data = open(r'..\..\runtime-rocm-cuda\ggml-hip.dll', 'rb').read()

# Search for the "hipMallocManaged unsupported" string and the code path around it
# This string indicates there's already a fallback path
target = b'hipMallocManaged unsupported'
idx = data.find(target)
print(f'String "{target.decode()}" at file offset 0x{idx:08X}')

# Also search for VMM-related strings
for s in [b'VMM', b'NO_VMM', b'use_vmm', b'hipMallocManaged', b'GGML_CUDA_ENABLE_UNIFIED_MEMORY']:
    idx2 = 0
    while True:
        idx2 = data.find(s, idx2)
        if idx2 == -1:
            break
        end = min(idx2 + 200, len(data))
        ctx = data[idx2:end].split(b'\x00')[0].decode('ascii', errors='replace')
        print(f'  "{s.decode()}" at 0x{idx2:08X}: "{ctx}"')
        idx2 += len(s)

# Check for GGML_HIP_NO_VMM or similar env var checks 
for s in [b'GGML_HIP', b'GGML_CUDA', b'HIP_VISIBLE', b'HSA_']:
    idx2 = 0
    while True:
        idx2 = data.find(s, idx2)
        if idx2 == -1:
            break
        end = min(idx2 + 200, len(data))
        ctx = data[idx2:end].split(b'\x00')[0].decode('ascii', errors='replace')
        print(f'  "{s.decode()}" at 0x{idx2:08X}: "{ctx}"')
        idx2 += len(s)
