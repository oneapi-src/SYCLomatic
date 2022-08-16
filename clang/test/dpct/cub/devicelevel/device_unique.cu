// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_unique %S/device_unique.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_unique/device_unique.dp.cpp --match-full-lines %s

// Missing wait() synchronization for memcpy with dependencies

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>

#define N 8

// CHECK:void test_1() {
// CHECK:dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:int h_in[N] = {0, 2, 2, 9, 5, 5, 5, 8};
// CHECK:int h_out[N] = {0};
// CHECK:int *d_in = nullptr;
// CHECK:int *d_out = nullptr;
// CHECK:int *d_selected_num = nullptr;
// CHECK:int h_selected_num = 0;
// CHECK:d_in = (int *)sycl::malloc_device(sizeof(h_in), q_ct1);
// CHECK:d_out = (int *)sycl::malloc_device(sizeof(h_out), q_ct1);
// CHECK:d_selected_num = sycl::malloc_device<int>(1, q_ct1);
// CHECK:q_ct1.memcpy((void *)d_in, (void *)h_in, sizeof(h_in)).wait();
// CHECK:DPCT1026{{.*}}
// CHECK:q_ct1.fill(d_selected_num, oneapi::dpl::unique_copy(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + N, d_out) - d_out, 1);
// CHECK:q_ct1.memcpy((void *)&h_selected_num, (void *)d_selected_num, sizeof(int)){{.*}};
// CHECK:q_ct1.memcpy((void *)h_out, (void *)d_out, h_selected_num * sizeof(int)).wait();
// CHECK:}
void test_1() {
  int h_in[N] = {0, 2, 2, 9, 5, 5, 5, 8};
  int h_out[N] = {0};
  int *d_in = nullptr;
  int *d_out = nullptr;
  int *d_temp = nullptr;
  int *d_selected_num = nullptr;
  int h_selected_num = 0;
  size_t d_temp_size = 0;
  cudaMalloc((void **)&d_in, sizeof(h_in));
  cudaMalloc((void **)&d_out, sizeof(h_out));
  cudaMalloc((void **)&d_selected_num, sizeof(int));
  cudaMemcpy((void *)d_in, (void *)h_in, sizeof(h_in), cudaMemcpyHostToDevice);
  cub::DeviceSelect::Unique(nullptr, d_temp_size, d_in, d_out, d_selected_num, N);
  cudaMalloc((void **)&d_temp, d_temp_size);
  cub::DeviceSelect::Unique((void *)d_temp, d_temp_size, d_in, d_out, d_selected_num, N);
  cudaMemcpy((void *)&h_selected_num, (void *)d_selected_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_out, (void *)d_out, h_selected_num * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", h_selected_num);
  for (int i = 0; i < h_selected_num; ++i)
    printf("%d\n", h_out[i]);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_temp);
  cudaFree(d_selected_num);
}

// CHECK:void test_2() {
// CHECK:dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:int h_in[N] = {0, 2, 2, 9, 5, 5, 5, 8};
// CHECK:int h_out[N] = {0};
// CHECK:int *d_in = nullptr;
// CHECK:int *d_out = nullptr;
// CHECK:int *d_selected_num = nullptr;
// CHECK:int h_selected_num = 0;
// CHECK:d_in = (int *)sycl::malloc_device(sizeof(h_in), q_ct1);
// CHECK:d_out = (int *)sycl::malloc_device(sizeof(h_out), q_ct1);
// CHECK:d_selected_num = sycl::malloc_device<int>(1, q_ct1);
// CHECK:q_ct1.memcpy((void *)d_in, (void *)h_in, sizeof(h_in)).wait();
// CHECK:DPCT1027:{{.*}}
// CHECK:0, 0;
// CHECK:q_ct1.fill(d_selected_num, oneapi::dpl::unique_copy(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + N, d_out) - d_out, 1);
// CHECK:q_ct1.memcpy((void *)&h_selected_num, (void *)d_selected_num, sizeof(int)){{.*}};
// CHECK:q_ct1.memcpy((void *)h_out, (void *)d_out, h_selected_num * sizeof(int)).wait();
// }
void test_2() {
  int h_in[N] = {0, 2, 2, 9, 5, 5, 5, 8};
  int h_out[N] = {0};
  int *d_in = nullptr;
  int *d_out = nullptr;
  int *d_temp = nullptr;
  int *d_selected_num = nullptr;
  int h_selected_num = 0;
  size_t d_temp_size = 0;
  cudaMalloc((void **)&d_in, sizeof(h_in));
  cudaMalloc((void **)&d_out, sizeof(h_out));
  cudaMalloc((void **)&d_selected_num, sizeof(int));
  cudaMemcpy((void *)d_in, (void *)h_in, sizeof(h_in), cudaMemcpyHostToDevice);
  cub::DeviceSelect::Unique(nullptr, d_temp_size, d_in, d_out, d_selected_num, N), 0;
  cudaMalloc((void **)&d_temp, d_temp_size);
  cub::DeviceSelect::Unique((void *)d_temp, d_temp_size, d_in, d_out, d_selected_num, N);
  cudaMemcpy((void *)&h_selected_num, (void *)d_selected_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_out, (void *)d_out, h_selected_num * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", h_selected_num);
  for (int i = 0; i < h_selected_num; ++i)
    printf("%d\n", h_out[i]);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_temp);
  cudaFree(d_selected_num);
}
