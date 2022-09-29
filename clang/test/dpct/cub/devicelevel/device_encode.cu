// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_encode %S/device_encode.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_encode/device_encode.dp.cpp --match-full-lines %s

// Missing wait() synchronization for memcpy with dependencies

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
// CHECK:#include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>

#define N 8

// CHECK: void test_1() {
// CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK: int h_in[N] = {0, 2, 2, 9, 5, 5, 5, 8};
// CHECK: int h_unique[N] = {0};
// CHECK: int h_counts[N] = {0};
// CHECK: int *d_in = nullptr;
// CHECK: int *d_unique = nullptr;
// CHECK: int *d_counts = nullptr;
// CHECK: int *d_selected_num = nullptr;
// CHECK: int h_selected_num = 0;
// CHECK: d_in = (int *)sycl::malloc_device(sizeof(h_in), q_ct1);
// CHECK: d_unique = (int *)sycl::malloc_device(sizeof(h_unique), q_ct1);
// CHECK: d_counts = (int *)sycl::malloc_device(sizeof(h_counts), q_ct1);
// CHECK: d_selected_num = sycl::malloc_device<int>(1, q_ct1);
// CHECK: q_ct1.memcpy((void *)d_in, (void *)h_in, sizeof(h_in)).wait();
// CHECK: DPCT1026:{{.*}}
// CHECK: q_ct1.fill(d_selected_num, std::distance(d_unique, oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + N, dpct::device_vector<size_t>(N, 1).begin(), d_unique, d_counts).first), 1).wait();
// CHECK: q_ct1.memcpy((void *)&h_selected_num, (void *)d_selected_num, sizeof(int)){{.*}};
// CHECK: q_ct1.memcpy((void *)h_unique, (void *)d_unique, h_selected_num * sizeof(int)){{.*}};
// CHECK: q_ct1.memcpy((void *)h_counts, (void *)d_counts, h_selected_num * sizeof(int)).wait();
// CHECK: }
void test_1() {
  int h_in[N] = {0, 2, 2, 9, 5, 5, 5, 8};
  int h_unique[N] = {0};
  int h_counts[N] = {0};
  int *d_in = nullptr;
  int *d_temp = nullptr;
  int *d_unique = nullptr;
  int *d_counts = nullptr;
  int *d_selected_num = nullptr;
  int h_selected_num = 0;
  size_t d_temp_size = 0;
  cudaMalloc((void **)&d_in, sizeof(h_in));
  cudaMalloc((void **)&d_unique, sizeof(h_unique));
  cudaMalloc((void **)&d_counts, sizeof(h_counts));
  cudaMalloc((void **)&d_selected_num, sizeof(int));
  cudaMemcpy((void *)d_in, (void *)h_in, sizeof(h_in), cudaMemcpyHostToDevice);
  cub::DeviceRunLengthEncode::Encode(nullptr, d_temp_size, d_in, d_unique, d_counts, d_selected_num, N);
  cudaMalloc((void **)&d_temp, d_temp_size);
  cub::DeviceRunLengthEncode::Encode(d_temp, d_temp_size, d_in, d_unique, d_counts, d_selected_num, N);
  cudaMemcpy((void *)&h_selected_num, (void *)d_selected_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_unique, (void *)d_unique, h_selected_num * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_counts, (void *)d_counts, h_selected_num * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", h_selected_num);
  for (int i = 0; i < h_selected_num; ++i)
    printf("%d ", h_unique[i]);
  printf("\n");
  for (int i = 0; i < h_selected_num; ++i)
    printf("%d ", h_counts[i]);
  printf("\n");
  cudaFree(d_in);
  cudaFree(d_unique);
  cudaFree(d_counts);
  cudaFree(d_temp);
  cudaFree(d_selected_num);
}

// CHECK: void test_2() {
// CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK: int h_in[N] = {0, 2, 2, 9, 5, 5, 5, 8};
// CHECK: int h_unique[N] = {0};
// CHECK: int h_counts[N] = {0};
// CHECK: int *d_in = nullptr;
// CHECK: int *d_unique = nullptr;
// CHECK: int *d_counts = nullptr;
// CHECK: int *d_selected_num = nullptr;
// CHECK: int h_selected_num = 0;
// CHECK: d_in = (int *)sycl::malloc_device(sizeof(h_in), q_ct1);
// CHECK: d_unique = (int *)sycl::malloc_device(sizeof(h_unique), q_ct1);
// CHECK: d_counts = (int *)sycl::malloc_device(sizeof(h_counts), q_ct1);
// CHECK: d_selected_num = sycl::malloc_device<int>(1, q_ct1);
// CHECK: q_ct1.memcpy((void *)d_in, (void *)h_in, sizeof(h_in)).wait();
// CHECK: DPCT1027:{{.*}}
// CHECK: 0, 0;
// CHECK: q_ct1.fill(d_selected_num, std::distance(d_unique, oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + N, dpct::device_vector<size_t>(N, 1).begin(), d_unique, d_counts).first), 1).wait();
// CHECK: q_ct1.memcpy((void *)&h_selected_num, (void *)d_selected_num, sizeof(int)){{.*}};
// CHECK: q_ct1.memcpy((void *)h_unique, (void *)d_unique, h_selected_num * sizeof(int)){{.*}};
// CHECK: q_ct1.memcpy((void *)h_counts, (void *)d_counts, h_selected_num * sizeof(int)).wait();
// CHECK: }
void test_2() {
  int h_in[N] = {0, 2, 2, 9, 5, 5, 5, 8};
  int h_unique[N] = {0};
  int h_counts[N] = {0};
  int *d_in = nullptr;
  int *d_temp = nullptr;
  int *d_unique = nullptr;
  int *d_counts = nullptr;
  int *d_selected_num = nullptr;
  int h_selected_num = 0;
  size_t d_temp_size = 0;
  cudaMalloc((void **)&d_in, sizeof(h_in));
  cudaMalloc((void **)&d_unique, sizeof(h_unique));
  cudaMalloc((void **)&d_counts, sizeof(h_counts));
  cudaMalloc((void **)&d_selected_num, sizeof(int));
  cudaMemcpy((void *)d_in, (void *)h_in, sizeof(h_in), cudaMemcpyHostToDevice);
  cub::DeviceRunLengthEncode::Encode(nullptr, d_temp_size, d_in, d_unique, d_counts, d_selected_num, N), 0;
  cudaMalloc((void **)&d_temp, d_temp_size);
  cub::DeviceRunLengthEncode::Encode(d_temp, d_temp_size, d_in, d_unique, d_counts, d_selected_num, N);
  cudaMemcpy((void *)&h_selected_num, (void *)d_selected_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_unique, (void *)d_unique, h_selected_num * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_counts, (void *)d_counts, h_selected_num * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", h_selected_num);
  for (int i = 0; i < h_selected_num; ++i)
    printf("%d ", h_unique[i]);
  printf("\n");
  for (int i = 0; i < h_selected_num; ++i)
    printf("%d ", h_counts[i]);
  printf("\n");
  cudaFree(d_in);
  cudaFree(d_unique);
  cudaFree(d_counts);
  cudaFree(d_temp);
  cudaFree(d_selected_num);
}
