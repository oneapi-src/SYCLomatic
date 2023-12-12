// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_merge_sort %S/device_merge_sort.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_merge_sort/device_merge_sort.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/devicelevel/device_merge_sort/device_merge_sort.dp.cpp -o %T/devicelevel/device_merge_sort/device_merge_sort.dp.o %}

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
// CHECK:#include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>

struct CustomOpT {
  template <typename DataType>
  bool operator()(const DataType &lhs, const DataType &rhs) {
    return lhs <= rhs;
  }
} op;

int n, num_items, *d_keys, *d_values, *d_outs;

// CHECK: void sort_pairs() {
void sort_pairs() {
  {
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op);
  }
  // CHECK: {
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortPairs was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: dpct::sort(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, d_values, op);
  // CHECK: }

  {
    cudaStream_t s = nullptr;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = &q_ct1;
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortPairs was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: dpct::sort(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, d_values, op);
  // CHECK: }

  {
    cudaStream_t s = (cudaStream_t)(intptr_t)0xFF;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = dpct::int_as_queue_ptr((intptr_t)0xFF);
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortPairs was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: dpct::sort(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, d_values, op);
  // CHECK: }
}

// CHECK: void stable_sort_pairs() {
void stable_sort_pairs() {
  {
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::StableSortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::StableSortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op);
  }
  // CHECK: {
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::StableSortPairs was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: dpct::stable_sort(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, d_values, op);
  // CHECK: }

  {
    cudaStream_t s = nullptr;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::StableSortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::StableSortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = &q_ct1;
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::StableSortPairs was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: dpct::stable_sort(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, d_values, op);
  // CHECK: }

  {
    cudaStream_t s = (cudaStream_t)(intptr_t)0xFF;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::StableSortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::StableSortPairs(temp_storage, temp_storage_size, d_keys, d_values, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = dpct::int_as_queue_ptr((intptr_t)0xFF);
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::StableSortPairs was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: dpct::stable_sort(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, d_values, op);
  // CHECK: }
}

// CHECK: void sort_keys() {
void sort_keys() {
  {
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortKeys(temp_storage, temp_storage_size, d_keys, num_items, op);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortKeys(temp_storage, temp_storage_size, d_keys, num_items, op);
  }
  // CHECK: {
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortKeys was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::sort(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, op);
  // CHECK: }

  {
    cudaStream_t s = nullptr;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortKeys(temp_storage, temp_storage_size, d_keys, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortKeys(temp_storage, temp_storage_size, d_keys, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = &q_ct1;
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortKeys was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::sort(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, op);
  // CHECK: }

  {
    cudaStream_t s = (cudaStream_t)(intptr_t)0xFF;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortKeys(temp_storage, temp_storage_size, d_keys, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortKeys(temp_storage, temp_storage_size, d_keys, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = dpct::int_as_queue_ptr((intptr_t)0xFF);
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortKeys was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::sort(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, op);
  // CHECK: }
}

// CHECK: void stable_sort_keys() {
void stable_sort_keys() {
  {
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::StableSortKeys(temp_storage, temp_storage_size, d_keys, num_items, op);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::StableSortKeys(temp_storage, temp_storage_size, d_keys, num_items, op);
  }
  // CHECK: {
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::StableSortKeys was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::stable_sort(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, op);
  // CHECK: }

  {
    cudaStream_t s = nullptr;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::StableSortKeys(temp_storage, temp_storage_size, d_keys, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::StableSortKeys(temp_storage, temp_storage_size, d_keys, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = &q_ct1;
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::StableSortKeys was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::stable_sort(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, op);
  // CHECK: }

  {
    cudaStream_t s = (cudaStream_t)(intptr_t)0xFF;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::StableSortKeys(temp_storage, temp_storage_size, d_keys, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::StableSortKeys(temp_storage, temp_storage_size, d_keys, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = dpct::int_as_queue_ptr((intptr_t)0xFF);
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::StableSortKeys was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::stable_sort(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, op);
  // CHECK: }
}

// CHECK: void sort_keys_copy() {
void sort_keys_copy() {
  {
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortKeysCopy(temp_storage, temp_storage_size, d_keys, d_outs, num_items, op);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortKeysCopy(temp_storage, temp_storage_size, d_keys, d_outs, num_items, op);
  }
  // CHECK: {
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortKeysCopy was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::partial_sort_copy(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, d_outs, d_outs + num_items, op);
  // CHECK: }

  {
    cudaStream_t s = nullptr;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortKeysCopy(temp_storage, temp_storage_size, d_keys, d_outs, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortKeysCopy(temp_storage, temp_storage_size, d_keys, d_outs, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = &q_ct1;
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortKeysCopy was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::partial_sort_copy(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, d_outs, d_outs + num_items, op);
  // CHECK: }

  {
    cudaStream_t s = (cudaStream_t)(intptr_t)0xFF;
    void *temp_storage = nullptr;
    size_t temp_storage_size;
    cub::DeviceMergeSort::SortKeysCopy(temp_storage, temp_storage_size, d_keys, d_outs, num_items, op, s);
    cudaMalloc(&temp_storage, temp_storage_size);
    cub::DeviceMergeSort::SortKeysCopy(temp_storage, temp_storage_size, d_keys, d_outs, num_items, op, s);
  }
  // CHECK: {
  // CHECK: dpct::queue_ptr s = dpct::int_as_queue_ptr((intptr_t)0xFF);
  // CHECK-NOT: void *temp_storage = nullptr;
  // CHECK-NOT: size_t temp_storage_size;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{.*}}: The call to cub::DeviceMergeSort::SortKeysCopy was removed because this functionality is redundant in SYCL.
  // CHECK: */
  // CHECK-NEXT: oneapi::dpl::partial_sort_copy(oneapi::dpl::execution::device_policy(*s), d_keys, d_keys + num_items, d_outs, d_outs + num_items, op);
  // CHECK: }
}
