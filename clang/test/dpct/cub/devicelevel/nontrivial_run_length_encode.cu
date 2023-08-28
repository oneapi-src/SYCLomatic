// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/nontrivial_run_length_encode %S/nontrivial_run_length_encode.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/nontrivial_run_length_encode/nontrivial_run_length_encode.dp.cpp --match-full-lines %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
// CHECK:#include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>

int num_items;
int *d_in;
int *d_offsets_out;
int *d_lengths_out;
int *d_num_runs_out;

void test1() {
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceRunLengthEncode::NonTrivialRuns was removed because this call is redundant in SYCL.
  cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes, d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: dpct::nontrivial_run_length_encode(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
  cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes, d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
}

void test2() {
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  // CHECK: DPCT1027:{{.*}}: The call to cub::DeviceRunLengthEncode::NonTrivialRuns was replaced with 0 because this call is redundant in SYCL.
  // CHECK: 0, 0;
  cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes, d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items), 0;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: dpct::nontrivial_run_length_encode(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
  cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes, d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
}

void test3() {
  cudaStream_t S = nullptr;
  cudaStreamCreate(&S);
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceRunLengthEncode::NonTrivialRuns was removed because this call is redundant in SYCL.
  cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes, d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: dpct::nontrivial_run_length_encode(oneapi::dpl::execution::device_policy(*S), d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
  cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes, d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items, S);
}
