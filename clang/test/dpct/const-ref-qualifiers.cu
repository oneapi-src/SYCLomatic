// RUN: dpct --usm-level=none -out-root %T/const-ref-qualifiers %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/const-ref-qualifiers/const-ref-qualifiers.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdio.h>
// CHECK-EMPTY:
// CHECK-NEXT: void foo2(const dpct::queue_ptr stream) {}
// CHECK-NEXT: void foo3(dpct::queue_ptr &stream) {}
// CHECK-NEXT: void foo4(const dpct::queue_ptr &stream) {}
// CHECK-NEXT: void foo5(dpct::queue_ptr *stream) {}
// CHECK-NEXT: void foo6(dpct::queue_ptr &&stream) {}
#include <stdio.h>

void foo2(const cudaStream_t stream) {}
void foo3(cudaStream_t &stream) {}
void foo4(const cudaStream_t &stream) {}
void foo5(cudaStream_t *stream) {}
void foo6(cudaStream_t &&stream) {}

// CHECK: #define CS dpct::queue_ptr
#define CS cudaStream_t

// TODO: migrate types in typedef and using
typedef cudaStream_t CS2;
using CS3 = cudaStream_t;

void bar() {
  // CHECK: dpct::queue_ptr streams[10];
  cudaStream_t streams[10];
  CS s;
  CS2 s2;
  CS3 s3;
}

