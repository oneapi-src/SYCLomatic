// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/key_value_pair %S/key_value_pair.cu --cuda-include-path="%cuda-path/include" -- -std=c++17 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/key_value_pair/key_value_pair.dp.cpp %s

// CHECK: #include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>

// CHECK: void foo(dpct:key_value_pair<int, int> sum) {}
__global__ void foo(cub::KeyValuePair<int, int> sum) {}

template<typename BinOp>
void foo1(BinOp op) {
  BinOp op2 = op;
  (void) op2;
}

template <typename T>
struct Foo {
  // CHECK: dpct:key_value_pair<T, T> pair; 
  cub::KeyValuePair<T, T> pair;
};

int test1() {
  // CHECK: dpct:key_value_pair<int, int> pair;
  cub::KeyValuePair<int, int> pair;

  // CHECK: dpct::get_default_queue().parallel_for(
  // CHECK:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK:     foo(dpct:key_value_pair<int, int>());
  // CHECK:   });
  foo<<<1, 1>>>(cub::KeyValuePair<int, int>());

  // CHECK: foo1<dpct:key_value_pair<int, int>>(max);
  foo1<cub::KeyValuePair<int, int>>(pair);

  return 0;
}
