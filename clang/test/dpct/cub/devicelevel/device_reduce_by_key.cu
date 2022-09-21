// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_reduce_by_key %S/device_reduce_by_key.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_reduce_by_key/device_reduce_by_key.dp.cpp %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cub/cub.cuh>
#include <iostream>

struct CustomMin {
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};


// CHECK:void test_1() {
// CHECK-NEXT:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK-NEXT:  sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK-NEXT:  int n = 8;
// CHECK-NEXT:  CustomMin op;
// CHECK-NEXT:  int unq[n], agg[n], num;
// CHECK-NEXT:  int key[] = {0, 2, 2, 9, 5, 5, 5, 8};
// CHECK-NEXT:  int val[] = {0, 7, 1, 6, 2, 5, 3, 4};
// CHECK-NEXT:  int *d_key, *d_val, *d_unq, *d_agg, *d_num;
// CHECK-NEXT:  d_key = (int *)sycl::malloc_device(sizeof(key), q_ct1);
// CHECK-NEXT:  d_val = (int *)sycl::malloc_device(sizeof(val), q_ct1);
// CHECK-NEXT:  d_unq = (int *)sycl::malloc_device(sizeof(key), q_ct1);
// CHECK-NEXT:  d_agg = (int *)sycl::malloc_device(sizeof(key), q_ct1);
// CHECK-NEXT:  d_num = sycl::malloc_device<int>(1, q_ct1);
// CHECK-NEXT:  q_ct1.memcpy(d_key, key, sizeof(key)){{.*}};
// CHECK-NEXT:  q_ct1.memcpy(d_val, val, sizeof(val)){{.*}};
// CHECK-NEXT:  DPCT1026{{.*}}
// CHECK-NEXT:  q_ct1.fill(d_num, std::distance(d_unq, oneapi::dpl::reduce_by_key(oneapi::dpl::execution::device_policy(q_ct1), d_key, d_key + n, d_val, d_unq, d_agg, std::equal_to<typename std::iterator_traits<decltype(d_key)>::value_type>(), op).first), 1).wait();
// CHECK-NEXT:  q_ct1.memcpy(&num, d_num, sizeof(int)){{.*}};
// CHECK-NEXT:  q_ct1.memcpy(unq, d_unq, sizeof(int) * num){{.*}};
// CHECK-NEXT:  q_ct1.memcpy(agg, d_agg, sizeof(int) * num){{.*}};
// CHECK-NEXT:}
void test_1() {
  int n = 8;
  CustomMin op;
  int unq[n], agg[n], num;
  int key[] = {0, 2, 2, 9, 5, 5, 5, 8};
  int val[] = {0, 7, 1, 6, 2, 5, 3, 4};
  int *d_key, *d_val, *d_unq, *d_agg, *d_num;
  cudaMalloc(&d_key, sizeof(key));
  cudaMalloc(&d_val, sizeof(val));
  cudaMalloc(&d_unq, sizeof(key));
  cudaMalloc(&d_agg, sizeof(key));
  cudaMalloc(&d_num, sizeof(int));
  cudaMemcpy(d_key, key, sizeof(key), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, val, sizeof(val), cudaMemcpyHostToDevice);
  void *tmp = nullptr;
  size_t n_tmp = 0;
  cub::DeviceReduce::ReduceByKey(tmp, n_tmp, d_key, d_unq, d_val, d_agg, d_num, op, n);
  cudaMalloc(&tmp, n_tmp);
  cub::DeviceReduce::ReduceByKey(tmp, n_tmp, d_key, d_unq, d_val, d_agg, d_num, op, n);
  cudaMemcpy(&num, d_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(unq, d_unq, sizeof(int) * num, cudaMemcpyDeviceToHost);
  cudaMemcpy(agg, d_agg, sizeof(int) * num, cudaMemcpyDeviceToHost);
}

// CHECK:void test_2() {
// CHECK-NEXT:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK-NEXT:  sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK-NEXT:  int n = 8;
// CHECK-NEXT:  CustomMin op;
// CHECK-NEXT:  int unq[n], agg[n], num;
// CHECK-NEXT:  int key[] = {0, 2, 2, 9, 5, 5, 5, 8};
// CHECK-NEXT:  int val[] = {0, 7, 1, 6, 2, 5, 3, 4};
// CHECK-NEXT:  int *d_key, *d_val, *d_unq, *d_agg, *d_num;
// CHECK-NEXT:  d_key = (int *)sycl::malloc_device(sizeof(key), q_ct1);
// CHECK-NEXT:  d_val = (int *)sycl::malloc_device(sizeof(val), q_ct1);
// CHECK-NEXT:  d_unq = (int *)sycl::malloc_device(sizeof(key), q_ct1);
// CHECK-NEXT:  d_agg = (int *)sycl::malloc_device(sizeof(key), q_ct1);
// CHECK-NEXT:  d_num = sycl::malloc_device<int>(1, q_ct1);
// CHECK-NEXT:  q_ct1.memcpy(d_key, key, sizeof(key)){{.*}};
// CHECK-NEXT:  q_ct1.memcpy(d_val, val, sizeof(val)){{.*}};
// CHECK-NEXT:  DPCT1026{{.*}}
// CHECK-NEXT:  0, 0;
// CHECK-NEXT:  q_ct1.fill(d_num, std::distance(d_unq, oneapi::dpl::reduce_by_key(oneapi::dpl::execution::device_policy(q_ct1), d_key, d_key + n, d_val, d_unq, d_agg, std::equal_to<typename std::iterator_traits<decltype(d_key)>::value_type>(), op).first), 1).wait();
// CHECK-NEXT:  q_ct1.memcpy(&num, d_num, sizeof(int)){{.*}};
// CHECK-NEXT:  q_ct1.memcpy(unq, d_unq, sizeof(int) * num){{.*}};
// CHECK-NEXT:  q_ct1.memcpy(agg, d_agg, sizeof(int) * num){{.*}};
// CHECK-NEXT:}
void test_2() {
  int n = 8;
  CustomMin op;
  int unq[n], agg[n], num;
  int key[] = {0, 2, 2, 9, 5, 5, 5, 8};
  int val[] = {0, 7, 1, 6, 2, 5, 3, 4};
  int *d_key, *d_val, *d_unq, *d_agg, *d_num;
  cudaMalloc(&d_key, sizeof(key));
  cudaMalloc(&d_val, sizeof(val));
  cudaMalloc(&d_unq, sizeof(key));
  cudaMalloc(&d_agg, sizeof(key));
  cudaMalloc(&d_num, sizeof(int));
  cudaMemcpy(d_key, key, sizeof(key), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, val, sizeof(val), cudaMemcpyHostToDevice);
  void *tmp = nullptr;
  size_t n_tmp = 0;
  cub::DeviceReduce::ReduceByKey(tmp, n_tmp, d_key, d_unq, d_val, d_agg, d_num, op, n), 0;
  cudaMalloc(&tmp, n_tmp);
  cub::DeviceReduce::ReduceByKey(tmp, n_tmp, d_key, d_unq, d_val, d_agg, d_num, op, n);
  cudaMemcpy(&num, d_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(unq, d_unq, sizeof(int) * num, cudaMemcpyDeviceToHost);
  cudaMemcpy(agg, d_agg, sizeof(int) * num, cudaMemcpyDeviceToHost);
}
