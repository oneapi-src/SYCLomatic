// RUN: dpct --format-range=none -out-root %T/histogram %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/histogram/histogram.dp.cpp %s
#include <cstdint>
#include <cuda_runtime.h>

__global__ void wg_private_local_kernel(const uint32_t N, const uint32_t B, const uint32_t* input, uint32_t* histogram)
{}

void foo() {
  const int B = 32;
  int grid_size, block_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:0: Migration of cudaOccupancyMaxPotentialBlockSize is not supported.
  // CHECK-NEXT: */
  cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, &wg_private_local_kernel, B * sizeof(uint32_t));

  // CHECK: dpct::device_info device_properties;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1019:{{[0-9a-z]+}}: local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. You may need to adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: size_t smem_size = device_properties.get_local_mem_size();
  cudaDeviceProp device_properties;
  size_t smem_size = device_properties.sharedMemPerBlock;
}

