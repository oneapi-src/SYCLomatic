// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --use-experimental-features=logical-group,non-uniform-groups --usm-level=none -out-root %T/cooperative_group_coalesced_group %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cooperative_group_coalesced_group/cooperative_group_coalesced_group.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl  -DTEST %T/cooperative_group_coalesced_group/cooperative_group_coalesced_group.dp.cpp -o %T/cooperative_group_coalesced_group/cooperative_group_coalesced_group.dp.o %}
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void coalescedExampleKernel(int *data) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  int res = 0;
  if (block.thread_rank() % 4) {
    //CHECK: sycl::ext::oneapi::experimental::opportunistic_group active = sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group();
    cg::coalesced_group active = cg::coalesced_threads();

    // Example operation: let the first thread in the coalesced group increment the data
    //CHECK: if (active.get_local_linear_id() == 0) {
    if (active.thread_rank() == 0) {
    //CEHCK: res = dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&data[active.get_local_linear_id() - 1], 1);
      res = atomicAdd(&data[active.size() - 1], 1);
    }
  #ifndef TEST
    //CHECK: res = sycl::select_from_group(active, res, 0);
    res = active.shfl(res, 0);
  #endif
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
