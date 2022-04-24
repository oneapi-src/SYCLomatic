// RUN: dpct --format-range=none --usm-level=none -out-root %T/api-name-migrated %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/api-name-migrated/api-name-migrated.dp.cpp %s

#include <cuda_runtime.h>

void fooo() {
  size_t size = 10 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;

  size_t length = size * size * size;
  size_t bytes = length * sizeof(float);
  float *src;

  // CHECK: free(d_A);
  cudaFreeHost(d_A);

  // CHECK: src = (float *)malloc(bytes);
  cudaMallocHost(&src, bytes);

  struct cudaPitchedPtr srcGPU;

  // CHECK: sycl::range<3> extent = sycl::range<3>(size * sizeof(float), size, size);
  struct cudaExtent extent = make_cudaExtent(size * sizeof(float), size, size);

  // CHECK: srcGPU = dpct::dpct_malloc(extent);
  cudaMalloc3D(&srcGPU, extent);
}

