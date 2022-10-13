// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_inclusive_sum %S/device_inclusive_sum.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_inclusive_sum/device_inclusive_sum.dp.cpp %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <cassert>

// CHECK: DPCT1026:{{.*}}
// CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, device_out);
void test_1() {
  int n = 10;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int host_out[10];
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cub::DeviceScan::InclusiveSum(nullptr, n_device_tmp, device_in, device_out, n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::InclusiveSum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
}

// CHECK: DPCT1027:{{.*}}
// CHECK-NEXT: 0, 0;
// CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, device_out);
void test_2() {
  int n = 10;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int host_out[10];
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cub::DeviceScan::InclusiveSum(nullptr, n_device_tmp, device_in, device_out, n), 0;
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::InclusiveSum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
}

// CHECK: dpct::queue_ptr stream = (dpct::queue_ptr)(void *)(uintptr_t)5;
// CHECK-NEXT: DPCT1026:{{.*}}
// CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(*stream), device_in, device_in + n, device_out);
void test_3() {
  int n = 10;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int host_out[10];
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cudaStream_t stream = (cudaStream_t)(void *)(uintptr_t)5;
  cub::DeviceScan::InclusiveSum(nullptr, n_device_tmp, device_in, device_out, n, stream);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::InclusiveSum((void *)device_tmp, n_device_tmp, device_in, device_out, n, stream);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
}

