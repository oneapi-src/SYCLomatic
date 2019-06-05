// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --match-full-lines --input-file %T/histogram.sycl.cpp %s

#include <cuda_runtime.h>

__global__ void wg_private_local_kernel(const uint32_t N, const uint32_t B, const uint32_t* input, uint32_t* histogram)
{}

void foo() {
  const int B = 32;
  int grid_size, block_size;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:0: cudaOccupancyMaxPotentialBlockSize is not supported in DPC++
  // CHECK-NEXT: */
  cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, &wg_private_local_kernel, B * sizeof(uint32_t));

  // CHECK: syclct::sycl_device_info device_properties;
  // CHECK-NEXT: /*
  // CHECK-NEXT: SYCLCT1019:{{[0-9a-z]+}}: The sharedMemPerBlock is not necessarily the same as local_mem_size in DPC++
  // CHECK-NEXT: */
  // CHECK-NEXT: size_t smem_size = device_properties.get_local_mem_size();
  cudaDeviceProp device_properties;
  size_t smem_size = device_properties.sharedMemPerBlock;
}
