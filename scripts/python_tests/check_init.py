data = open(r'..\..\runtime-rocm-cuda\ggml-hip.dll', 'rb').read()
for s in [b'ggml_cuda_init', b'PEER_MAX_BATCH', b'hipSetDevice', b'hipDeviceReset', 
          b'hipDeviceSetLimit', b'hipDeviceSetCacheConfig', b'hipDeviceSetMemPool',
          b'hipMemPoolCreate', b'hipMemPool', b'hipDeviceSet']:
    idx = data.find(s)
    if idx != -1:
        end = min(idx + 150, len(data))
        ctx = data[idx:end].split(b'\x00')[0].decode('ascii', errors='replace')
        print(f'{s.decode()}: at 0x{idx:08X}: {ctx}')
    else:
        print(f'{s.decode()}: NOT FOUND')
