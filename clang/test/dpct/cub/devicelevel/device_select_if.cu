// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_select_flagged %S/device_select_flagged.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_select_flagged/device_select_flagged.dp.cpp --match-full-lines %s

// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
// CHECK: #include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>
#include <vector>

struct LessThan {
  int compare;
  inline LessThan(int compare) : compare(compare) {}
  __device__ bool operator()(const int &a) const {
    return (a < compare);
  }
};

int num_items;           // e.g., 8
int *d_in;               // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
int *d_out;              // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
int *d_num_selected_out; // e.g., [ ]
LessThan select_op(7);

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMalloc(&p, sizeof(T) * list.size());
  cudaMemcpy(p, list.begin(), sizeof(T) * list.size(), cudaMemcpyHostToDevice);
  return p;
}

void test1() {
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);
  cudaFree(d_temp_storage);
}
// CHECK: void test1() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: DPCT1026:{{.*}}: The call to cub::DeviceSelect::If was removed because this call is redundant in SYCL.
// CHECK: q_ct1.fill(d_num_selected_out, std::distance(d_out, oneapi::dpl::copy_if(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + num_items, d_out, select_op)), 1).wait();
// CHECK-NOT: sycl::free(d_temp_storage, q_ct1);
// CHECK: }

void test2() {
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  auto res = cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);
  cudaFree(d_temp_storage);
}
// CHECK: void test2() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: DPCT1027:{{.*}}: The call to cub::DeviceSelect::If was replaced with 0 because this call is redundant in SYCL.
// CHECK: auto res = 0;
// CHECK: q_ct1.fill(d_num_selected_out, std::distance(d_out, oneapi::dpl::copy_if(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + num_items, d_out, select_op)), 1).wait();
// CHECK-NOT: sycl::free(d_temp_storage, q_ct1);
// CHECK: }

void test3() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, s);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
}
// CHECK: void test3() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK; DPCT1026:{{.*}}: The call to cub::DeviceSelect::If was removed because this call is redundant in SYCL.
// CHECK: s->fill(d_num_selected_out, std::distance(d_out, oneapi::dpl::copy_if(oneapi::dpl::execution::device_policy(*s), d_in, d_in + num_items, d_out, select_op)), 1).wait();
// CHECK: }

void test4() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  auto res = cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, s);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
}
// CHECK: void test4() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK; DPCT1027:{{.*}}: The call to cub::DeviceSelect::If was replaced with 0 because this call is redundant in SYCL.
// CHECK: auto res = 0;
// CHECK: s->fill(d_num_selected_out, std::distance(d_out, oneapi::dpl::copy_if(oneapi::dpl::execution::device_policy(*s), d_in, d_in + num_items, d_out, select_op)), 1).wait();
// CHECK: }

int main() {
  num_items = 8;
  d_in = init({0, 2, 3, 9, 5, 2, 81, 8});
  d_out = init({0, 0, 0, 0, 0, 0, 0, 0});
  d_num_selected_out = init({0});
  test1();
  test2();
  test3();
  test4();
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_num_selected_out);
  return 0;
}
