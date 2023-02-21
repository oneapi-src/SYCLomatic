// RUN: dpct --format-range=none -out-root %T/math-function-cl %s --cuda-include-path="%cuda-path/include" --no-cl-namespace-inline --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/math-function-cl/math-function-cl.dp.cpp --match-full-lines %s

#include <cmath>

#include <math_functions.h>

__device__ float4 fun() {
  float4 a, b, c;
  // CHECK: cl::sycl::fma(a[0], b[0], c[0]);
  __fmaf_rn(a.x, b.x, c.x);
  // CHECK: return cl::sycl::mfloat4(cl::sycl::fma(a[0], b[0], c[0]), cl::sycl::fma(a[1], b[1], c[1]), cl::sycl::fma(a[2], b[2], c[2]), cl::sycl::fma(a[3], b[3], c[3]));
  return make_float4(__fmaf_rd(a.x, b.x, c.x), __fmaf_rz(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w));
}


__global__ void kernel() {

}

void foo() {
  // CHECK:   dpct::get_default_queue().parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, ceil(2.3)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[0-9a-z]+}}) {
  // CHECK-NEXT:           kernel();
  // CHECK-NEXT:         });
  kernel<<< ceil(2.3), 1 >>>();
}

int main() {

}

