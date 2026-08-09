#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    printf("Device: %s\n", props.name);
    printf("Total Global Mem: %.2f GiB\n", props.totalGlobalMem / (1024.0*1024*1024));
    printf("isLargeBar: %d\n", props.isLargeBar);
    printf("isIntegrated: %d\n", props.integrated);
    
    // Try allocating 80 GB
    size_t alloc_size = (size_t)80 * 1024 * 1024 * 1024;
    void* ptr = nullptr;
    printf("\nTrying hipMalloc(%.1f GiB)...\n", alloc_size / (1024.0*1024*1024));
    hipError_t err = hipMalloc(&ptr, alloc_size);
    if (err == hipSuccess) {
        printf("SUCCESS! Allocated %.1f GiB at %p\n", alloc_size / (1024.0*1024*1024), ptr);
        hipFree(ptr);
    } else {
        printf("FAILED: %s (%d)\n", hipGetErrorString(err), err);
    }
    
    // Try 60 GB
    alloc_size = (size_t)60 * 1024 * 1024 * 1024;
    printf("\nTrying hipMalloc(%.1f GiB)...\n", alloc_size / (1024.0*1024*1024));
    err = hipMalloc(&ptr, alloc_size);
    if (err == hipSuccess) {
        printf("SUCCESS! Allocated %.1f GiB at %p\n", alloc_size / (1024.0*1024*1024), ptr);
        hipFree(ptr);
    } else {
        printf("FAILED: %s (%d)\n", hipGetErrorString(err), err);
    }

    // Try 40 GB
    alloc_size = (size_t)40 * 1024 * 1024 * 1024;
    printf("\nTrying hipMalloc(%.1f GiB)...\n", alloc_size / (1024.0*1024*1024));
    err = hipMalloc(&ptr, alloc_size);
    if (err == hipSuccess) {
        printf("SUCCESS! Allocated %.1f GiB at %p\n", alloc_size / (1024.0*1024*1024), ptr);
        hipFree(ptr);
    } else {
        printf("FAILED: %s (%d)\n", hipGetErrorString(err), err);
    }
    
    return 0;
}
