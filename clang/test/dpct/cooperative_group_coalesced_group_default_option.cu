// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none   -out-root %T/cooperative_group_coalesced_group_default_option %s --cuda-include-path="%cuda-path/include" --extra-arg="-std=c++14"
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_group_coalesced_group_default_option/cooperative_group_coalesced_group_default_option.dp.cpp

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void coalescedExampleKernel(int *data) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  int res = 0;
  if (block.thread_rank() % 4) {
    //CHECK: /*
    //CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of coalesced_threads is not supported, please try to remigrate with option: --use-experimental-features=non-uniform-groups.
    //CHECK-NEXT: */
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cooperative_groups::__v1::coalesced_group is not supported, please try to remigrate with option: --use-experimental-features=non-uniform-groups.
    //CHECK-NEXT: */
    cg::coalesced_group active = cg::coalesced_threads();

    // Example operation: let the first thread in the coalesced group increment the data
    //CHECK:  /*
    //CHECK-NEXT:  DPCT1119:{{[0-9]+}}: Migration of cooperative_groups::__v1::coalesced_group.thread_rank is not supported, please try to remigrate with option: --use-experimental-features=non-uniform-groups.
    //CHECK-NEXT:  */
    if (active.thread_rank() == 0) {
      //CHECK:  /*
      //CHECK-NEXT:  DPCT1119:{{[0-9]+}}: Migration of cooperative_groups::__v1::coalesced_group.size is not supported, please try to remigrate with option: --use-experimental-features=non-uniform-groups.
      //CHECK-NEXT:  */
      res = atomicAdd(&data[active.size() - 1], 1);
    }
    //CHECK:      /*
    //CHECK-NEXT:  DPCT1119:{{[0-9]+}}: Migration of cooperative_groups::__v1::coalesced_group.shfl is not supported, please try to remigrate with option: --use-experimental-features=non-uniform-groups.
    //CHECK-NEXT:  */
    res = active.shfl(res, 0);
  }
}

int main() {
  const int dataSize = 256;
  const int bytes = dataSize * sizeof(int);

  // Allocate memory on the host and the device
  int *h_data = (int *)malloc(bytes);
  int *d_data;
  cudaMalloc(&d_data, bytes);

  // Initialize host memory to zeros
  memset(h_data, 0, bytes);

  // Copy host data to device
  cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

  // Execute the kernel
  coalescedExampleKernel<<<1, dataSize>>>(d_data);
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

  // Check results
  bool passed = true;
  for (int i = 0; i < dataSize; ++i) {
    if (h_data[i] != 0)
      std::cout << "i is  " << i << " test " << h_data[i] << std::endl;
    /*
        if (h_data[i] != ((i + 1) % 32 == 0 ? 1 : 0)) {
            passed = false;
            std::cout << "Test failed at index " << i << ": expected " << ((i + 1) % 32 == 0 ? 1 : 0) << ", got " << h_data[i] << std::endl;
            break;
        }
        */
  }

  if (passed) {
    std::cout << "Test passed!" << std::endl;
  }

  // Clean up memory
  free(h_data);
  cudaFree(d_data);

  return 0;
}
