// RUN: syclct -out-root %T %s -passes "MemoryTranslationRule" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memory_management.sycl.cpp %s

#include <cuda_runtime.h>

void fooo() {
    size_t size = 1234567 * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *d_A = NULL;
    // CHECK: (syclct::sycl_malloc<char>((void **)&d_A, size), 0);
    cudaMalloc((void **)&d_A, size);
    // CHECK: (syclct::sycl_memset((void*)(d_A), (void*)(15), (void*)(size)), 0);
    cudaMemset(d_A, 0xf, size);
    // CHECK: (syclct::sycl_memcpy<char>(d_A, h_A, size, cudaMemcpyHostToDevice), 0);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    // CHECK: (syclct::sycl_memcpy<char>(h_A, d_A, size, cudaMemcpyDeviceToHost), 0);
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    // CHECK: (syclct::sycl_free<char>(d_A), 0);
    cudaFree(d_A);
    free(h_A);
}

cudaError_t mallocWrapper(void **buffer, size_t size) {
  // CHECK: return (syclct::sycl_malloc<char>(buffer, size), 0);
  return cudaMalloc(buffer, size);
}