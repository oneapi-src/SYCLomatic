// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/null_type %S/null_type.cu --cuda-include-path="%cuda-path/include" -- -std=c++17 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/null_type/null_type.dp.cpp %s

// CHECK: #include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>

// CHECK: void foo(dpct:null_type dummy) {}
__global__ void foo(cub::NullType dummy) {}

template <typename T>
void foo1(T op) {
  T op2 = op;
  (void)op2;
}

int test1() {
  // CHECK: dpct:null_type pair;
  cub::NullType null;

  // CHECK: dpct::get_in_order_queue().parallel_for(
  // CHECK:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK:     foo(dpct:null_type());
  // CHECK:   });
  foo<<<1, 1>>>(cub::NullType());

  // CHECK: foo1<dpct:null_type>(null);
  foo1<cub::NullType>(null);

  return 0;
}
