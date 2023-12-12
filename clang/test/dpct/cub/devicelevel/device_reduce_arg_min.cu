// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_reduce_arg_min %S/device_reduce_arg_min.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_reduce_arg_min/device_reduce_arg_min.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/devicelevel/device_reduce_arg_min/device_reduce_arg_min.dp.cpp -o %T/devicelevel/device_reduce_arg_min/device_reduce_arg_min.dp.o %}

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cub/cub.cuh>
#include <initializer_list>
#include <cstddef>

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMalloc<T>(&p, sizeof(T) * list.size());
  cudaMemcpy(p, list.begin(), sizeof(T) * list.size(), cudaMemcpyHostToDevice);
  return p;
}

int num_items = 7;
int *d_in;
 cub::KeyValuePair<int, int> *d_out;

void test1() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaFree(d_temp_storage);
}
// CHECK: void test1() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: DPCT1026:{{.*}}: The call to cub::DeviceReduce::ArgMin was removed because this functionality is redundant in SYCL.
// CHECK: dpct::reduce_argmin(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_out, num_items);
// CHECK-NOT: sycl::free({{.*}})
// CHECK: }

void test2() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto res = cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaFree(d_temp_storage);
}
// CHECK: void test2() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: DPCT1027:{{.*}}: The call to cub::DeviceReduce::ArgMin was replaced with 0 because this functionality is redundant in SYCL.
// CHECK: auto res = 0;
// CHECK: dpct::reduce_argmin(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_out, num_items);
// CHECK-NOT: sycl::free({{.*}})
// CHECK: }

void test3() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaStream_t s;
  cudaStreamCreate(&s);
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, s);
  cudaStreamDestroy(s);
  cudaFree(d_temp_storage);
}
// CHECK: void test3() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: dpct::queue_ptr s;
// CHECK: s = dev_ct1.create_queue();
// CHECK: DPCT1026:{{.*}}: The call to cub::DeviceReduce::ArgMin was removed because this functionality is redundant in SYCL.
// CHECK: dpct::reduce_argmin(oneapi::dpl::execution::device_policy(*s), d_in, d_out, num_items);
// CHECK: dev_ct1.destroy_queue(s);
// CHECK-NOT: sycl::free({{.*}})
// CHECK: }

int main() {
  d_in = init({8, 6, 7, 5, 3, 0, 9});
  d_out = init<cub::KeyValuePair<int, int>>({{-1, -1}});
  test1();
  test2();
  test3();
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
