// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_radix_sort %S/device_radix_sort.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/devicelevel/device_radix_sort/device_radix_sort.dp.cpp

#include <cub/cub.cuh>

int n, *d_keys_in, *d_keys_out, *d_values_in, *d_values_out;
// CHECK: dpct::io_iterator_pair<int *> key_buffers(d_keys_in, d_keys_out);
cub::DoubleBuffer<int> key_buffers(d_keys_in, d_keys_out);
// CHECK: dpct::io_iterator_pair<int *> value_buffers(d_values_in, d_values_out);
cub::DoubleBuffer<int> value_buffers(d_values_in, d_values_out);

// CHECK: DPCT1082:{{.*}}: Migration of cub::DoubleBuffer<int[2]> type is not supported.
// CHECK: cub::DoubleBuffer<int[2]> test;
cub::DoubleBuffer<int[2]> test;

// CHECK: DPCT1082:{{.*}}: Migration of cub::DoubleBuffer<int (*)[2]> type is not supported.
// CHECK: cub::DoubleBuffer<int (*)[2]> test_1;
cub::DoubleBuffer<int (*)[2]> test_1;

// CHECK: dpct::io_iterator_pair<int * *> test_2;
cub::DoubleBuffer<int *> test_2;

// CHECK: auto double_buffer(dpct::io_iterator_pair<int *> &buffers) {
auto double_buffer(cub::DoubleBuffer<int> &buffers) {
  // CHECK: buffers.selector = !buffers.selector;
  buffers.selector = !buffers.selector;
  // CHECK: buffers.iter[0] += 1;
  buffers.d_buffers[0] += 1;
  // CHECK: *buffers.iter[1] = 5;
  *buffers.d_buffers[1] = 5;
  // CHECK: buffers = dpct::io_iterator_pair<int *>(buffers.second(), buffers.first());
  buffers = cub::DoubleBuffer<int>(buffers.Alternate(), buffers.Current());
  // CHECK: buffers = dpct::io_iterator_pair<int *>();
  buffers = cub::DoubleBuffer<int>();
}

void test1(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, false);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size, d_keys_in, d_keys_out, n);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, n, false, true);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size, key_buffers, n);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, key_buffers, n);
  // end
}

void test2(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, false, 1);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, 1);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, n, false, true, 1);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size, key_buffers, n, 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, key_buffers, n, 1);
  // end
}

void test3(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, false, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, 1, 4);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, 1, 4);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, n, false, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size, key_buffers, n, 1, 4);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, key_buffers, n, 1, 4);
  // end
}

void test4(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end

  cudaStream_t s = 0;

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_out, n, false, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, 1, 4, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, 1, 4, s);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(*s), key_buffers, n, false, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size, key_buffers, n, 1, 4, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, key_buffers, n, 1, 4, s);
  // end
}

void test5(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, true);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, n);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, n, true, true);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_storage_size, key_buffers, n);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size, key_buffers, n);
  // end
}

void test6(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, true, 1);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, 1);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, n, true, true, 1);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_storage_size, key_buffers, n, 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size, key_buffers, n, 1);
  // end
}

void test7(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, 1, 4);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, 1, 4);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, n, true, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_storage_size, key_buffers, n, 1, 4);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size, key_buffers, n, 1, 4);
  // end
}

void test8(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end

  cudaStream_t s = 0;

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_out, n, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, n, 1, 4, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, n, 1, 4, s);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_keys(oneapi::dpl::execution::device_policy(*s), key_buffers, n, true, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_storage_size, key_buffers, n, 1, 4, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size, key_buffers, n, 1, 4, s);
  // end
}

void test9(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, false);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, value_buffers, n, false, true);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, key_buffers, value_buffers, n);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, key_buffers, value_buffers, n);
  // end
}

void test10(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, false, 1);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, value_buffers, n, false, true, 1);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, key_buffers, value_buffers, n, 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, key_buffers, value_buffers, n, 1);
  // end
}

void test11(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, false, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1, 4);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1, 4);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, value_buffers, n, false, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, key_buffers, value_buffers, n, 1, 4);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, key_buffers, value_buffers, n, 1, 4);
  // end
}

void test12(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end

  cudaStream_t s = 0;

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_out, d_values_in, d_values_out, n, false, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1, 4, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1, 4, s);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(*s), key_buffers, value_buffers, n, false, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, key_buffers, value_buffers, n, 1, 4, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, key_buffers, value_buffers, n, 1, 4, s);
  // end
}

void test13(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, true);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, value_buffers, n, true, true);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_size, key_buffers, value_buffers, n);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_size, key_buffers, value_buffers, n);
  // end
}

void test14(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, true, 1);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, value_buffers, n, true, true, 1);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_size, key_buffers, value_buffers, n, 1);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_size, key_buffers, value_buffers, n, 1);
  // end
}

void test15(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1, 4);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1, 4);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), key_buffers, value_buffers, n, true, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_size, key_buffers, value_buffers, n, 1, 4);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_size, key_buffers, value_buffers, n, 1, 4);
  // end
}

void test16(void) {
  // CHECK: // begin
  // CHECK-NOT: void *temp_storage;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: // end  
  // begin
  void *temp_storage;
  size_t temp_storage_size;
  // end  

  cudaStream_t s = 0;
  
  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_out, d_values_in, d_values_out, n, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1, 4, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_size, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 1, 4, s);
  // end

  // CHECK: // begin
  // CHECK: DPCT1026:{{.*}}
  // CHECK-NOT: cudaMalloc(&temp_storage, temp_storage_size);
  // CHECK: dpct::sort_pairs(oneapi::dpl::execution::device_policy(*s), key_buffers, value_buffers, n, true, true, 1, 4);
  // CHECK: // end
  // begin
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_size, key_buffers, value_buffers, n, 1, 4, s);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_size, key_buffers, value_buffers, n, 1, 4, s);
  // end
}
