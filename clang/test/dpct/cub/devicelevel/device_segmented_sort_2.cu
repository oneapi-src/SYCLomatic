// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_segmented_sort_2 %S/device_segmented_sort_2.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/devicelevel/device_segmented_sort_2/device_segmented_sort_2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/devicelevel/device_segmented_sort_2/device_segmented_sort_2.dp.cpp -o %T/devicelevel/device_segmented_sort_2/device_segmented_sort_2.dp.o %}

#include <cub/cub.cuh>

int num_items, num_segments, *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_offsets;
// CHECK: dpct::io_iterator_pair<int *> d_keys(d_keys_in, d_keys_out);
cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
// CHECK: dpct::io_iterator_pair<int *> d_values(d_values_in, d_values_out);
cub::DoubleBuffer<int> d_values(d_values_in, d_values_out);

void test1(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, false, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1);
  // end
}

void test2(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, false, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, 0);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, 0);
  // end
}

void test3(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  cudaStream_t stream;

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, stream);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, stream);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(*stream), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, false, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, stream);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, stream);
  // end
}

void test4(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, false);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, false);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, false, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, 0, false);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, 0, false);
  // end
}

void test5(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  cudaStream_t stream;

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, stream, false);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, stream, false);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(*stream), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, false, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairs(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, stream, false);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, stream, false);
  // end
}


void test6(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, true, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1);
  // end
}

void test7(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, true, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, 0);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, 0);
  // end
}

void test8(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  cudaStream_t stream;

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, stream);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, stream);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(*stream), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, true, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, stream);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, stream);
  // end
}

void test9(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, false);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, false);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, true, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, 0, false);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, 0, false);
  // end
}

void test10(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *d_temp_storage;
  size_t temp_storage_bytes;
  // end

  cudaStream_t stream;

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, stream, false);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, stream, false);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK: dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(*stream), d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, true, true);
  // CHECK: // end
  // begin
  cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, stream, false);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_offsets, d_offsets + 1, stream, false);
  // end
}
