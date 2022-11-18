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

// CHECK: DPCT1026:{{.*}}
// CHECK: q_ct1.fill(d_selected_num, std::distance(d_unique, oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + N, dpct::constant_iterator<size_t>(1), d_unique, d_counts).first), 1).wait();
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

// CHECK: DPCT1027:{{.*}}
// CHECK: 0, 0;
// CHECK: q_ct1.fill(d_selected_num, std::distance(d_unique, oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + N, dpct::constant_iterator<size_t>(1), d_unique, d_counts).first), 1).wait();
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

// CHECK: dpct::queue_ptr stream = (dpct::queue_ptr)(void *)(uintptr_t)5;
// CHECK: DPCT1026:{{.*}}
// CHECK: stream->fill(d_selected_num, std::distance(d_unique, oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + N, dpct::constant_iterator<size_t>(1), d_unique, d_counts).first), 1).wait();
void test_3() {
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
  cudaStream_t stream = (cudaStream_t)(void *)(uintptr_t)5;
  cub::DeviceRunLengthEncode::Encode(nullptr, d_temp_size, d_in, d_unique, d_counts, d_selected_num, N, stream);
  cudaMalloc((void **)&d_temp, d_temp_size);
  cub::DeviceRunLengthEncode::Encode(d_temp, d_temp_size, d_in, d_unique, d_counts, d_selected_num, N, stream);
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
