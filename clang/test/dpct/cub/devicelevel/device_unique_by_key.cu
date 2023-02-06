// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.4, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.4, v11.8
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_unique_by_key %S/device_unique_by_key.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_unique_by_key/device_unique_by_key.dp.cpp %s

// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
// CHECK: #include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>
#include <vector>

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMalloc(&p, sizeof(T) * list.size());
  cudaMemcpy(p, list.begin(), sizeof(T) * list.size(), cudaMemcpyHostToDevice);
  return p;
}

int  num_items;              // e.g., 8
int  *d_keys_in;             // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
int  *d_values_in;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
int  *d_keys_out;            // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
int  *d_values_out;          // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
int  *d_num_selected_out;    // e.g., [ ]

void test1() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items);
  cudaFree(d_temp_storage);
}
// CHECK: void test1() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: DPCT1026:{{.*}}: The call to cub::DeviceSelect::UniqueByKey was removed because this call is redundant in SYCL.
// CHECK: q_ct1.fill(d_num_selected_out, std::distance(d_keys_out, std::get<0>(dpct::unique_copy(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_in + num_items, d_values_in, d_keys_out, d_values_out))), 1).wait();
// CHECK-NOT: sycl::free(d_temp_storage, q_ct1);
// CHECK: }

void test2() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto res = cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items);
  cudaFree(d_temp_storage);
}
// CHECK: void test2() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: DPCT1027:{{.*}}: The call to cub::DeviceSelect::UniqueByKey was replaced with 0 because this call is redundant in SYCL.
// CHECK: q_ct1.fill(d_num_selected_out, std::distance(d_keys_out, std::get<0>(dpct::unique_copy(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_in + num_items, d_values_in, d_keys_out, d_values_out))), 1).wait();
// CHECK-NOT: sycl::free(d_temp_storage, q_ct1);
// CHECK: }

void test3() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items, s);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
}
// CHECK: void test3() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK; DPCT1026:{{.*}}: The call to cub::DeviceSelect::UniqueByKey was removed because this call is redundant in SYCL.
// CHECK: s->fill(d_num_selected_out, std::distance(d_keys_out, std::get<0>(dpct::unique_copy(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_in + num_items, d_values_in, d_keys_out, d_values_out))), 1).wait();
// CHECK: }

void test4() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto res = cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items, s);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
}
// CHECK: void test4() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK; DPCT1027:{{.*}}: The call to cub::DeviceSelect::UniqueByKey was replaced with 0 because this call is redundant in SYCL.
// CHECK: s->fill(d_num_selected_out, std::distance(d_keys_out, std::get<0>(dpct::unique_copy(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_in + num_items, d_values_in, d_keys_out, d_values_out))), 1).wait();
// CHECK: }

int main() {
  num_items = 8;
  d_keys_in = init({0, 2, 2, 9, 5, 5, 5, 8});
  d_values_in = init({1, 2, 3, 4, 5, 6, 7, 8});
  d_keys_out = init({0, 0, 0, 0, 0, 0, 0, 0});
  d_values_out = init({0, 0, 0, 0, 0, 0, 0, 0});
  d_num_selected_out = init({0});
  test1();
  test2();
  test3();
  test4();
  cudaFree(d_keys_in);
  cudaFree(d_values_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_out);
  cudaFree(d_num_selected_out);
  return 0;
}
