// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_partition %S/device_partition.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_partition/device_partition.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/devicelevel/device_partition/device_partition.dp.cpp -o %T/devicelevel/device_partition/device_partition.dp.o %}

// Missing wait() synchronization for memcpy with dependencies

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
// CHECK:#include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>
// clang-format off
int num_items = 8;
int *d_in;
int *d_flags;
int *d_out;
int *d_num_selected_out;
int *d_large_out;
int *d_small_out;
int *d_unselected_out;

struct LessThan {
  int compare;
  explicit LessThan(int compare) : compare(compare) {}
  __device__ bool operator()(const int &a) const { return (a < compare); }
};

// Functor type for selecting values greater than some criteria
struct GreaterThan {
  int compare;
  explicit GreaterThan(int compare) : compare(compare) {}
  __device__ bool operator()(const int &a) const { return a > compare; }
};

LessThan select_op(7), small_items_selector(7);
GreaterThan large_items_selector(50);

void test() {
  {
    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    // CHECK: DPCT1026:0: The call to cub::DevicePartition::Flagged was removed because this functionality is redundant in SYCL.
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // CHECK: dpct::partition_flagged(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_flags, d_out, d_num_selected_out, num_items);
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
  }
  {
    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    // CHECK: DPCT1026:1: The call to cub::DevicePartition::Flagged was removed because this functionality is redundant in SYCL.
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, s);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // CHECK: dpct::partition_flagged(oneapi::dpl::execution::device_policy(*s), d_in, d_flags, d_out, d_num_selected_out, num_items);
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, s);
  }
  {
    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    // CHECK: DPCT1026:2: The call to cub::DevicePartition::If was removed because this functionality is redundant in SYCL.
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // CHECK: dpct::partition_if(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_out, d_num_selected_out, num_items, select_op);
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);
  }
  {
    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    // CHECK: DPCT1026:3: The call to cub::DevicePartition::If was removed because this functionality is redundant in SYCL.
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, s);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // CHECK: dpct::partition_if(oneapi::dpl::execution::device_policy(*s), d_in, d_out, d_num_selected_out, num_items, select_op);
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, s);
  }
  {
    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    // CHECK: DPCT1026:4: The call to cub::DevicePartition::If was removed because this functionality is redundant in SYCL.
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_large_out, d_small_out, d_unselected_out, d_num_selected_out, num_items, large_items_selector, small_items_selector);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // CHECK: dpct::partition_if(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_large_out, d_small_out, d_unselected_out, d_num_selected_out, num_items, large_items_selector, small_items_selector, false);
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_large_out, d_small_out, d_unselected_out, d_num_selected_out, num_items, large_items_selector, small_items_selector);
  }
  {
    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    // CHECK: DPCT1026:5: The call to cub::DevicePartition::If was removed because this functionality is redundant in SYCL.
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_large_out, d_small_out, d_unselected_out, d_num_selected_out, num_items, large_items_selector, small_items_selector, s);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // CHECK: dpct::partition_if(oneapi::dpl::execution::device_policy(*s), d_in, d_large_out, d_small_out, d_unselected_out, d_num_selected_out, num_items, large_items_selector, small_items_selector, false);
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_large_out, d_small_out, d_unselected_out, d_num_selected_out, num_items, large_items_selector, small_items_selector, s);
  }
}
// clang-format on
