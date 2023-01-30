// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_reduce_max %S/device_reduce_max.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_reduce_max/device_reduce_max.dp.cpp %s

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
int *d_out;

void test1() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaFree(d_temp_storage);
}
// CHECK: void test1() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: DPCT1026:{{.*}}: The call to cub::DeviceReduce::Max was removed because this call is redundant in SYCL.
// CHECK: q_ct1.fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + num_items, typename std::iterator_traits<decltype(d_out)>::value_type{}, sycl::maximum<>()), 1).wait();
// CHECK-NOT: sycl::free({{.*}})
// CHECK: }

void test2() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto res = cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaFree(d_temp_storage);
}
// CHECK: void test2() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: DPCT1027:{{.*}}: The call to cub::DeviceReduce::Max was replaced with 0 because this call is redundant in SYCL.
// CHECK: auto res = 0;
// CHECK: q_ct1.fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + num_items, typename std::iterator_traits<decltype(d_out)>::value_type{}, sycl::maximum<>()), 1).wait(); 
// CHECK-NOT: sycl::free({{.*}})
// CHECK: }

void test3() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaStream_t s;
  cudaStreamCreate(&s);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, s);
  cudaStreamDestroy(s);
  cudaFree(d_temp_storage);
}
// CHECK: void test3() {
// CHECK-NOT: void *d_temp_storage = NULL;
// CHECK-NOT: size_t temp_storage_bytes = 0;
// CHECK: dpct::queue_ptr s;
// CHECK: s = dev_ct1.create_queue();
// CHECK: DPCT1026:{{.*}}: The call to cub::DeviceReduce::Max was removed because this call is redundant in SYCL.
// CHECK: s->fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(*s), d_in, d_in + num_items, typename std::iterator_traits<decltype(d_out)>::value_type{}, sycl::maximum<>()), 1).wait();
// CHECK: dev_ct1.destroy_queue(s);
// CHECK-NOT: sycl::free({{.*}})
// CHECK: }

int main() {
  d_in = init({8, 6, 7, 5, -3, 0, 9});
  d_out = init({0});
  test1();
  test2();
  test3();
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}