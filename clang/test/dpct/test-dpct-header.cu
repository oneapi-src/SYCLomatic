// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only -DCUDA
// RUN: FileCheck --match-full-lines --input-file %T/test-dpct-header.dp.cpp %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "inc/header.h"
#include "inc/header.h"

// CHECK: #ifdef CUDA
// CHECK-NEXT: void foo() {
// CHECK-NEXT: }
#ifdef CUDA
__global__ void foo() {
}
#elif defined(OPENMP)
void foo() {
}
#else
void foo() {
}
#endif

int main() {
#ifdef CUDA
  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class foo_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           foo();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }

  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class bar_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           bar();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  foo<<<1, 1, 1>>>();
  bar<<<1, 1, 1>>>();
#else
  foo();
  bar();
#endif
}
