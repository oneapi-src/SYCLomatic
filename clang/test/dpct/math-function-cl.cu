// RUN: dpct --format-range=none --out-root %T/math-function-cl %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/math-function-cl/math-function-cl.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/math-function-cl/math-function-cl.dp.cpp -o %T/math-function-cl/math-function-cl.dp.o %}

#include <cmath>

#include <math_functions.h>

__device__ float4 fun() {
  float4 a, b, c;
  // CHECK: sycl::fma(a.x(), b.x(), c.x());
  __fmaf_rn(a.x, b.x, c.x);
  // CHECK: return sycl::float4(sycl::fma(a.x(), b.x(), c.x()), sycl::fma(a.y(), b.y(), c.y()), sycl::fma(a.z(), b.z(), c.z()), sycl::fma(a.w(), b.w(), c.w()));
  return make_float4(__fmaf_rd(a.x, b.x, c.x), __fmaf_rz(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w));
}


__global__ void kernel() {

}

void foo() {
  // CHECK:   dpct::get_in_order_queue().parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, ceil(2.3)), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {
  // CHECK-NEXT:           kernel();
  // CHECK-NEXT:         });
  kernel<<< ceil(2.3), 1 >>>();
}

int main() {
  return 0;
}
