// RUN: dpct --format-range=none -out-root %T/est-dpct-header %s %S/test-dpct-header-dup.cu --cuda-include-path="%cuda-path/include" --sycl-named-lambda -extra-arg="-I%S/inc" -- -std=c++14 -x cuda --cuda-host-only -DCUDA
// RUN: FileCheck --match-full-lines --input-file %T/est-dpct-header/test-dpct-header.dp.cpp %s

// RUN: FileCheck --match-full-lines --input-file %T/est-dpct-header/inc/header.tpp %S/inc/header.tpp
// RUN: FileCheck --match-full-lines --input-file %T/est-dpct-header/inc/header2.TPP %S/inc/header2.TPP

// RUN: FileCheck --match-full-lines --input-file %T/est-dpct-header/inc/header.inl %S/inc/header.inl
// RUN: FileCheck --match-full-lines --input-file %T/est-dpct-header/inc/header2.INL %S/inc/header2.INL

// RUN: FileCheck --match-full-lines --input-file %T/est-dpct-header/inc/header.inc %S/inc/header.inc
// RUN: FileCheck --match-full-lines --input-file %T/est-dpct-header/inc/header2.INC %S/inc/header2.INC

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

// CHECK: #include "inc/header3.c.dp.cpp"
// CHECK-NEXT: #include "inc/header3.c.dp.cpp"
// CHECK-NEXT: #include "inc/header4.c"
#include "inc/header3.c"
#include <inc/header3.c>
#include "inc/header4.c"

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
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
#ifdef CUDA
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class foo_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           foo();
  // CHECK-NEXT:         });

  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class bar_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           bar();
  // CHECK-NEXT:         });
  foo<<<1, 1, 1>>>();
  bar<<<1, 1, 1>>>();
#else
  foo();
  bar();
#endif
}
