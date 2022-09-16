#ifndef CLANG_DPCT_TEST_ONE_DPL_ALGO_EXEC_HEADER_INSERT_H
#define CLANG_DPCT_TEST_ONE_DPL_ALGO_EXEC_HEADER_INSERT_H


// CHECK: #define DUMMY_MACRO
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdio.h>
#define DUMMY_MACRO
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>

inline int ExclusiveSum() {
  int n = 10;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int host_out[10];
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cub::DeviceScan::ExclusiveSum(device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::ExclusiveSum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
  return host_out[0];
}

#endif // CLANG_DPCT_TEST_ONE_DPL_ALGO_EXEC_HEADER_INSERT_H
