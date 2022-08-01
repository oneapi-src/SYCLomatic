// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_exclusive_sum %S/device_exclusive_sum.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_exclusive_sum/device_exclusive_sum.dp.cpp --match-full-lines %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>

// void test_1() {
// dpct::device_ext &dev_ct1 = dpct::get_current_device();
// sycl::queue &q_ct1 = dev_ct1.default_queue();
// int n = 10;
// int *device_in = nullptr;
// int *device_out = nullptr;
// int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
// int host_out[10];
// device_in = sycl::malloc_device<int>(n, q_ct1);
// device_out = sycl::malloc_device<int>(n, q_ct1);
// q_ct1.memcpy(device_in, (void *)host_in, sizeof(host_in)).wait();
// DPCT1026:{{.*}}
// oneapi::dpl::exclusive_scan(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, device_out, 0);
// q_ct1.memcpy((void *)host_out, (void *)device_out, sizeof(host_out)).wait();
// sycl::free(device_in, q_ct1);
// sycl::free(device_out, q_ct1);
// }
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
  cub::DeviceScan::ExclusiveSum(device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::ExclusiveSum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
}

// void test_2() {
// dpct::device_ext &dev_ct1 = dpct::get_current_device();
// sycl::queue &q_ct1 = dev_ct1.default_queue();
// int n = 10;
// int *device_in = nullptr;
// int *device_out = nullptr;
// int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
// int host_out[10];
// device_in = sycl::malloc_device<int>(n, q_ct1);
// device_out = sycl::malloc_device<int>(n, q_ct1);
// q_ct1.memcpy(device_in, (void *)host_in, sizeof(host_in)).wait();
// DPCT1026:{{.*}}
// DPCT1027:{{.*}}
// 0, 0;
// oneapi::dpl::exclusive_scan(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, device_out, 0);
// q_ct1.memcpy((void *)host_out, (void *)device_out, sizeof(host_out)).wait();
// sycl::free(device_in, q_ct1);
// sycl::free(device_out, q_ct1);
// }
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
  cub::DeviceScan::ExclusiveSum(device_tmp, n_device_tmp, device_in, device_out, n), 0;
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::ExclusiveSum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
}
