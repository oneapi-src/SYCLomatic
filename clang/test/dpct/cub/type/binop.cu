// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/binop %S/binop.cu --cuda-include-path="%cuda-path/include" -- -std=c++17 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/binop/binop.dp.cpp %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cub/cub.cuh>

// CHECK: void foo(sycl::plus<> sum) {}
__global__ void foo(cub::Sum sum) {}

template<typename BinOp>
void foo1(BinOp op) {
  BinOp op2 = op;
}

int main() {
  // CHECK: sycl::plus<> sum;
  cub::Sum sum;

  // CHECK: sycl::maximum<> max;
  cub::Max max;

  // CHECK: sycl::minimum<> min;
  cub::Min min;

  // CHECK: std::equal_to<> eq;
  cub::Equality eq;

  // CHECK: dpct::get_default_queue().parallel_for(
  // CHECK:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK:     foo(sycl::plus<>());
  // CHECK:   });
  foo<<<1, 1>>>(cub::Sum());

  // CHECK: foo1<sycl::maximum<>>(max);
  foo1<cub::Max>(max);

  // CHECK: foo1<sycl::minimum<>>(min);
  foo1<cub::Min>(min);

  // CHECK: foo1<std::equal_to<>>(eq);
  foo1<cub::Equality>(eq);
  return 0;
}
