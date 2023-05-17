// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_segmented_sort_keys %S/device_segmented_sort_keys.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_segmented_sort_keys/device_segmented_sort_keys.dp.cpp %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
// CHECK:#include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>

int n, num_segments, *d_keys_in, *d_keys_out, *d_offsets;

// CHECK:dpct::io_iterator_pair<int *> d_keys(d_keys_in, d_keys_out);
cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);

void test1(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1);
  // CHECK: void test1(void)
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false);
}

void test2(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeys(temp_storage, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1);
  // CHECK: void test2(void)
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false, true);
}

void test3(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cudaStream_t s = 0;
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeys(temp_storage, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1, s);
  // CHECK: void test3(void)
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false, true);
}

void test4(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cudaStream_t s = 0;
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, s);
  // CHECK: void test4(void)
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::queue_ptr s = &q_ct1;
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false);
}

void test5(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cub::DeviceSegmentedSort::SortKeysDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1);
  // CHECK: void test5(void)
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true);
}

void test6(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cub::DeviceSegmentedSort::SortKeysDescending(nullptr, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1);
  // CHECK: void test6(void)
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true, true);
}

void test7(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cudaStream_t s = 0;
  cub::DeviceSegmentedSort::SortKeysDescending(nullptr, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1, s);
  // CHECK: void test7(void)
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true, true);
}

void test8(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cudaStream_t s = 0;
  cub::DeviceSegmentedSort::SortKeysDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, s);
  // CHECK: void test8(void)
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::queue_ptr s = &q_ct1;
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true);
}
