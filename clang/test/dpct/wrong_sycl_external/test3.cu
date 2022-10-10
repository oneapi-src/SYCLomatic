// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T/wrong_sycl_external/test3_out %S/test3.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %S/test3.cu --match-full-lines --input-file %T/wrong_sycl_external/test3_out/test3.dp.cpp
// RUN: FileCheck %S/test3.cuh --match-full-lines --input-file %T/wrong_sycl_external/test3_out/test3.dp.hpp

#include "test3.cuh"

  // CHECK: template <typename T> void do_work() {
template <typename T> __global__ void do_work() {
  T x=T(), y=T();
  // CHECK: func_wrapper<T>([] (T a, T b) -> T { return a * b; }).reduce(x, y);
  func_wrapper<T>([] __host__ __device__ (T a, T b) -> T { return a * b; }).reduce(x, y);
  // CHECK: func_wrapper<T>([] (T a, T b) { return a * b; }).reduce(x, y);
  func_wrapper<T>([] __host__ __device__ (T a, T b) { return a * b; }).reduce(x, y);
}

// CHECK: void kernel() { dpct::get_default_queue().parallel_for(
// CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
// CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:     do_work<int>();
// CHECK-NEXT:   }); }
void kernel() { do_work<int><<<16, 32>>>(); }