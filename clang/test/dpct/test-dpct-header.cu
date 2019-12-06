// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only -DCUDA
// RUN: FileCheck --match-full-lines --input-file %T/test-dpct-header.dp.cpp %s

// RUN: FileCheck --match-full-lines --input-file %T/inc/header.tpp %S/inc/header.tpp
// RUN: FileCheck --match-full-lines --input-file %T/inc/header2.TPP %S/inc/header2.TPP

// RUN: FileCheck --match-full-lines --input-file %T/inc/header.inl %S/inc/header.inl
// RUN: FileCheck --match-full-lines --input-file %T/inc/header2.INL %S/inc/header2.INL

// RUN: FileCheck --match-full-lines --input-file %T/inc/header.inc %S/inc/header.inc
// RUN: FileCheck --match-full-lines --input-file %T/inc/header2.INC %S/inc/header2.INC

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "inc/header.h"
#include "inc/header.h"
#include "inc/header.inl"
#include "inc/header.inc"
#include "inc/header2.INL"
#include "inc/header2.INC"
#include "inc/header.tpp"
#include "inc/header2.TPP"

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
  // CHECK:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class foo_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           foo();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });

  // CHECK:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class bar_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           bar();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  foo<<<1, 1, 1>>>();
  bar<<<1, 1, 1>>>();
#else
  foo();
  bar();
#endif
}
