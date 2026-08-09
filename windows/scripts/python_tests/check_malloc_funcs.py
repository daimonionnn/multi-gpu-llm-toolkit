import struct

data = open(r'..\..\runtime-rocm-cuda\ggml-hip.dll', 'rb').read()

# Search for HIP API function names in the import table or strings
funcs = [b'hipMalloc', b'hipMemAlloc', b'hipMallocManaged', b'hipMallocPitch', 
         b'hipHostMalloc', b'hipExtMallocWithFlags', b'hipMemAllocHost']

for func in funcs:
    idx = 0
    count = 0
    while True:
        idx = data.find(func, idx)
        if idx == -1:
            break
        # Check if it's a clean string (surrounded by \x00 or other control chars)
        end = data.index(b'\x00', idx) if b'\x00' in data[idx:idx+100] else idx+len(func)
        full = data[idx:end]
        count += 1
        idx = end
    if count > 0:
        print(f'Found "{func.decode()}" {count} times')
        # Show first few occurrences
        idx2 = 0
        for i in range(min(3, count)):
            idx2 = data.find(func, idx2)
            end = data.index(b'\x00', idx2) if b'\x00' in data[idx2:idx2+100] else idx2+len(func)
            full = data[idx2:end].decode('ascii', errors='replace')
            print(f'  at 0x{idx2:08X}: "{full}"')
            idx2 = end
