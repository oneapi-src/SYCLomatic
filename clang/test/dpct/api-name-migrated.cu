// RUN: dpct --usm-level=none -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --match-full-lines --input-file %T/api-name-migrated.dp.cpp %s

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

  // CHECK: *(&src) = malloc(bytes);
  cudaMallocHost(&src, bytes);

  struct cudaPitchedPtr srcGPU;

  // CHECK: /*
  // CHECK-NEXT:DPCT1007:{{[0-9]+}}: make_cudaExtent: Migration of this API is not supported.
  // CHECK-NEXT:*/
  struct cudaExtent extent = make_cudaExtent(size * sizeof(float), size, size);

  // CHECK: /*
  // CHECK-NEXT:DPCT1007:{{[0-9]+}}: cudaMalloc3D: Migration of this API is not supported.
  // CHECK-NEXT:*/
  cudaMalloc3D(&srcGPU, extent);
}
