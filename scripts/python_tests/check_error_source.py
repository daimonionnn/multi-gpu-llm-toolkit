import struct

# Check both DLLs for the error string
for name, path in [
    ('ggml-hip.dll', r'..\..\runtime-rocm-cuda\ggml-hip.dll'),
    ('ggml-cuda.dll', r'..\..\runtime-rocm-cuda\ggml-cuda.dll'),
]:
    data = open(path, 'rb').read()
    # Search for the exact error message
    for s in [b'cudaMalloc failed', b'allocating %.2f MiB', b'allocating %', b'on device %d']:
        idx = data.find(s)
        if idx != -1:
            end = min(idx + 200, len(data))
            ctx = data[idx:end].split(b'\x00')[0].decode('ascii', errors='replace')
            print(f'{name}: found "{s.decode()}" at 0x{idx:08X}: "{ctx}"')
        else:
            print(f'{name}: "{s.decode()}" NOT FOUND')
    print()
