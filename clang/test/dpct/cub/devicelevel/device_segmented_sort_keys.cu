// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_segmented_sort_keys %S/device_segmented_sort_keys.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_segmented_sort_keys/device_segmented_sort_keys.dp.cpp %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
// CHECK:#include <dpct/dpl_utils.hpp>
#include <cstdint>
#include <cub/cub.cuh>

int n, num_segments, *d_keys_in, *d_keys_out, *d_offsets;

// CHECK:dpct::io_iterator_pair<int *> d_keys(d_keys_in, d_keys_out);
cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);

// CHECK: void test1(void) {
// CHECK:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:   sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:   /*
// CHECK:   DPCT1026:0: The call to cub::DeviceSegmentedSort::SortKeys was removed because this call is redundant in SYCL.
// CHECK:   */
// CHECK:   dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false);
// CHECK: }
void test1(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1);
}

// CHECK: void test2(void) {
// CHECK:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:   sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:   /*
// CHECK:   DPCT1026:1: The call to cub::DeviceSegmentedSort::SortKeys was removed because this call is redundant in SYCL.
// CHECK:   */
// CHECK:   dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys, n, num_segments, d_offsets, d_offsets + 1, false, true);
// CHECK: }
void test2(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeys(temp_storage, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1);
}

// CHECK: void test3(void) {
// CHECK:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:   sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:   dpct::queue_ptr s = &q_ct1;
// CHECK:   /*
// CHECK:   DPCT1026:2: The call to cub::DeviceSegmentedSort::SortKeys was removed because this call is redundant in SYCL.
// CHECK:   */
// CHECK:   dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(*s), d_keys, n, num_segments, d_offsets, d_offsets + 1, false, true);
// CHECK: }
void test3(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cudaStream_t s = 0;
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeys(temp_storage, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1, s);
}

// CHECK: void test4(void) {
// CHECK:   dpct::queue_ptr s = (dpct::queue_ptr)(void *)(uintptr_t)0xFF;
// CHECK:   /*
// CHECK:   DPCT1026:3: The call to cub::DeviceSegmentedSort::SortKeys was removed because this call is redundant in SYCL.
// CHECK:   */
// CHECK:   dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false);
// CHECK: }
void test4(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cudaStream_t s = (cudaStream_t)(void *)(uintptr_t)0xFF;
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, s);
}

// CHECK: void test5(void) {
// CHECK:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:   sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:   /*
// CHECK:   DPCT1026:4: The call to cub::DeviceSegmentedSort::SortKeysDescending was removed because this call is redundant in SYCL.
// CHECK:   */
// CHECK:   dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true);
// CHECK: }
void test5(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cub::DeviceSegmentedSort::SortKeysDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1);
}

// CHECK: void test6(void) {
// CHECK:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:   sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:   /*
// CHECK:   DPCT1026:5: The call to cub::DeviceSegmentedSort::SortKeysDescending was removed because this call is redundant in SYCL.
// CHECK:   */
// CHECK:   dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys, n, num_segments, d_offsets, d_offsets + 1, true, true);
// CHECK: }
void test6(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cub::DeviceSegmentedSort::SortKeysDescending(nullptr, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1);
}

// CHECK: void test7(void) {
// CHECK:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:   sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:   dpct::queue_ptr s = &q_ct1;
// CHECK:   /*
// CHECK:   DPCT1026:6: The call to cub::DeviceSegmentedSort::SortKeysDescending was removed because this call is redundant in SYCL.
// CHECK:   */
// CHECK:   dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(*s), d_keys, n, num_segments, d_offsets, d_offsets + 1, true, true);
// CHECK: }
void test7(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cudaStream_t s = 0;
  cub::DeviceSegmentedSort::SortKeysDescending(nullptr, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys, n, num_segments, d_offsets, d_offsets + 1, s);
}

// CHECK: void test8(void) {
// CHECK:   dpct::queue_ptr s = (dpct::queue_ptr)(void *)(uintptr_t)0xFF;;
// CHECK:   /*
// CHECK:   DPCT1026:7: The call to cub::DeviceSegmentedSort::SortKeysDescending was removed because this call is redundant in SYCL.
// CHECK:   */
// CHECK:   dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true);
// CHECK: }
void test8(void) {
  void *temp_storage;
  size_t temp_storage_size;
  cudaStream_t s = (cudaStream_t)(void *)(uintptr_t)0xFF;
  ;
  cub::DeviceSegmentedSort::SortKeysDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceSegmentedSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, s);
}
