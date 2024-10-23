// RUN: dpct --out-root %T/user_defined_rule_helper1 %s --cuda-include-path="%cuda-path/include" --rule-file %S/xpu_1.yaml --format-range=none
// RUN: FileCheck --input-file %T/user_defined_rule_helper1/user_defined_rule_helper1.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/user_defined_rule_helper1/user_defined_rule_helper1.dp.cpp -o %T/user_defined_rule_helper1/user_defined_rule_helper1.dp.o %}

#ifndef NO_BUILD_TEST

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "xpu_helper1.h"
#include <cuda_runtime.h>

__global__ void foo1_kernel() {}
void foo1() {
  // CHECK: static_cast<sycl::queue&>(c10::xpu::getCurrentXPUStream1()).parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:     foo1_kernel();
  // CHECK-NEXT:   });
  foo1_kernel<<<1, 1>>>();
}

__global__ void foo2_kernel(double *d) {}

void foo2() {
  double *d;
  // CHECK: d = sycl::malloc_device<double>(1, static_cast<sycl::queue&>(c10::xpu::getCurrentXPUStream1()));
  // CHECK-NEXT: {
  // CHECK-NEXT:   dpct::has_capability_or_fail(static_cast<sycl::queue&>(c10::xpu::getCurrentXPUStream1()).get_device(), {sycl::aspect::fp64});
  // CHECK-EMPTY:
  // CHECK-NEXT:   static_cast<sycl::queue&>(c10::xpu::getCurrentXPUStream1()).parallel_for(
  // CHECK-NEXT:     sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:     [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:       foo2_kernel(d);
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  // CHECK-NEXT: dpct::dpct_free(d, static_cast<sycl::queue&>(c10::xpu::getCurrentXPUStream1()));
  cudaMalloc(&d, sizeof(double));
  foo2_kernel<<<1, 1>>>(d);
  cudaFree(d);
}

#endif
