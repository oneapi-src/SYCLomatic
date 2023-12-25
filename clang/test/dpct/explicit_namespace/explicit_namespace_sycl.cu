// RUN: dpct --format-range=none -in-root %S -out-root %T/explicit_namespace_sycl %S/explicit_namespace_sycl.cu --cuda-include-path="%cuda-path/include" --use-explicit-namespace=sycl --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/explicit_namespace_sycl/explicit_namespace_sycl.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/explicit_namespace_sycl/explicit_namespace_sycl.dp.cpp -o %T/explicit_namespace_sycl/explicit_namespace_sycl.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: using namespace dpct;
#include <cmath>
#include <cuda_runtime.h>


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
// CHECK:   get_in_order_queue().parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, ceil(2.3)), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {
// CHECK-NEXT:           kernel();
// CHECK-NEXT:         });
  kernel<<< ceil(2.3), 1 >>>();
}

int main() {

}