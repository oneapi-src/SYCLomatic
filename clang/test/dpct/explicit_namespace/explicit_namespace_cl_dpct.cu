// RUN: dpct --format-range=none -in-root %S -out-root %T/explicit_namespace_cl_dpct %S/explicit_namespace_cl_dpct.cu --cuda-include-path="%cuda-path/include" --use-explicit-namespace=cl,dpct --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/explicit_namespace_cl_dpct/explicit_namespace_cl_dpct.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/explicit_namespace_cl_dpct/explicit_namespace_cl_dpct.dp.cpp -o %T/explicit_namespace_cl_dpct/explicit_namespace_cl_dpct.dp.o %}

#ifndef BUILD_TEST
// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cmath>
#include <cuda_runtime.h>


__device__ float4 fun() {
  float4 a, b, c;
// CHECK: cl::sycl::fma(a.x(), b.x(), c.x());
  __fmaf_rn(a.x, b.x, c.x);
// CHECK: return cl::sycl::float4(cl::sycl::fma(a.x(), b.x(), c.x()), cl::sycl::fma(a.y(), b.y(), c.y()), cl::sycl::fma(a.z(), b.z(), c.z()), cl::sycl::fma(a.w(), b.w(), c.w()));
  return make_float4(__fmaf_rd(a.x, b.x, c.x), __fmaf_rz(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w));
}


__global__ void kernel() {

}

void foo() {
// CHECK:   dpct::get_in_order_queue().parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, ceil(2.3)), cl::sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[0-9a-z]+}}) {
// CHECK-NEXT:           kernel();
// CHECK-NEXT:         });
  kernel<<< ceil(2.3), 1 >>>();
}

int main() {

}
#endif