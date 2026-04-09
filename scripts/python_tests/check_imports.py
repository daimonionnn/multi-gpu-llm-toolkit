import struct

data = open(r'..\..\runtime-rocm-cuda\ggml-hip.dll', 'rb').read()
pe_off = struct.unpack_from('<I', data, 0x3C)[0]

opt_off = pe_off + 24
opt_magic = struct.unpack_from('<H', data, opt_off)[0]
if opt_magic == 0x20b:
    import_rva = struct.unpack_from('<I', data, opt_off + 120)[0]
    import_size = struct.unpack_from('<I', data, opt_off + 124)[0]
    print(f'Import table RVA: 0x{import_rva:X}, size: {import_size}')

for pattern in [b'amdhip', b'hiprtc', b'rocblas', b'rocsolver']:
    idx = 0
    while True:
        idx = data.find(pattern, idx)
        if idx == -1:
            break
        end = data.index(b'\x00', idx)
        s = data[idx:end].decode('ascii', errors='replace')
        print(f'Found: "{s}" at offset 0x{idx:X}')
        idx = end
