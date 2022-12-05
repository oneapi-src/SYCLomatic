// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-op %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-op/thrust-op.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
// CHECK-NEXT: #include <functional>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/functional.h>

int main() {
  // CHECK: dpct::device_vector<int> vec(13);
  thrust::device_vector<int> vec(13);
  // CHECK: dpct::device_vector<int>::iterator iter1 = vec.begin();
  thrust::device_vector<int>::iterator iter1 = vec.begin();
  // CHECK: dpct::device_vector<int>::iterator iter2 = iter1 + 7;
  thrust::device_vector<int>::iterator iter2 = iter1 + 7;
  // CHECK:int d = oneapi::dpl::distance(iter1, iter2);
  int d = thrust::distance(iter1, iter2);
  // CHECK: std::greater_equal<float> ge;
  thrust::greater_equal<float> ge;
  // CHECK: std::less_equal<float> le;
  thrust::less_equal<float> le;
  // CHECK: std::logical_and<float> la;
  thrust::logical_and<float> la;
  // CHECK: std::bit_and<float> ba;
  thrust::bit_and<float> ba;
  // CHECK: std::bit_or<float> bo;
  thrust::bit_or<float> bo;
  // CHECK: oneapi::dpl::minimum<float> m;
  thrust::minimum<float> m;
  // CHECK: std::bit_xor<float> bx;
  thrust::bit_xor<float> bx;
  return 0;
}