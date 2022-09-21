// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_reduce_sum %S/device_reduce_sum.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_reduce_sum/device_reduce_sum.dp.cpp --match-full-lines %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <cassert>

// CHECK:void test_1() {
// CHECK:dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:int n = 10;
// CHECK:int *device_in = nullptr;
// CHECK:int *device_out = nullptr;
// CHECK:int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
// CHECK:int host_out[10];
// CHECK:device_in = sycl::malloc_device<int>(n, q_ct1);
// CHECK:device_out = sycl::malloc_device<int>(n, q_ct1);
// CHECK:q_ct1.memcpy(device_in, (void *)host_in, sizeof(host_in)).wait();
// CHECK:DPCT1026{{.*}}
// CHECK:q_ct1.fill(device_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, std::iterator_traits<decltype(device_out)>::value_type()), 1).wait();
// CHECK:q_ct1.memcpy((void *)host_out, (void *)device_out, sizeof(host_out)).wait();
// CHECK:sycl::free(device_in, q_ct1);
// CHECK:sycl::free(device_out, q_ct1);
// CHECK:assert(host_out[0] == 55);
// CHECK:}
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
  cub::DeviceReduce::Sum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceReduce::Sum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
  assert(host_out[0] == 55);
}

// CHECK:void test_2() {
// CHECK:dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:int n = 10;
// CHECK:int *device_in = nullptr;
// CHECK:int *device_out = nullptr;
// CHECK:int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
// CHECK:int host_out[10];
// CHECK:device_in = sycl::malloc_device<int>(n, q_ct1);
// CHECK:device_out = sycl::malloc_device<int>(n, q_ct1);
// CHECK:q_ct1.memcpy(device_in, (void *)host_in, sizeof(host_in)).wait();
// CHECK:DPCT1026:{{.*}}
// CHECK:DPCT1026:{{.*}}
// CHECK:DPCT1026:{{.*}}
// CHECK:q_ct1.fill(device_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, std::iterator_traits<decltype(device_out)>::value_type()), 1).wait();
// CHECK:q_ct1.memcpy((void *)host_out, (void *)device_out, sizeof(host_out)).wait();
// CHECK:sycl::free(device_in, q_ct1);
// CHECK:sycl::free(device_out, q_ct1);
// CHECK:assert(host_out[0] == 55);
// CHECK:}
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
  cub::DeviceReduce::Sum(0, n_device_tmp, device_in, device_out, n);
  cub::DeviceReduce::Sum(NULL, n_device_tmp, device_in, device_out, n);
  cub::DeviceReduce::Sum(nullptr, n_device_tmp, device_in, device_out, n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceReduce::Sum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
  assert(host_out[0] == 55);
}

// CHECK:void test_3() {
// CHECK:dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:int n = 10;
// CHECK:int *device_in = nullptr;
// CHECK:int *device_out = nullptr;
// CHECK:int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
// CHECK:int host_out[10];
// CHECK:device_in = sycl::malloc_device<int>(n, q_ct1);
// CHECK:device_out = sycl::malloc_device<int>(n, q_ct1);
// CHECK:q_ct1.memcpy(device_in, (void *)host_in, sizeof(host_in)).wait();
// CHECK:DPCT1027{{.*}}
// CHECK:q_ct1.fill(device_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, std::iterator_traits<decltype(device_out)>::value_type()), 1).wait();
// CHECK:q_ct1.memcpy((void *)host_out, (void *)device_out, sizeof(host_out)).wait();
// CHECK:sycl::free(device_in, q_ct1);
// CHECK:sycl::free(device_out, q_ct1);
// CHECK:assert(host_out[0] == 55);
// CHECK:}
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
  cub::DeviceReduce::Sum(nullptr, n_device_tmp, device_in, device_out, n), 0;
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceReduce::Sum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
  assert(host_out[0] == 55);
}
