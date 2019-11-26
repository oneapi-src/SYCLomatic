// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/const-ref-qualifiers.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdio.h>
// CHECK-EMPTY:
// CHECK-NEXT: using queue_p = cl::sycl::queue *;
// CHECK-EMPTY:
// CHECK-NEXT: void foo2(const queue_p stream) {}
// CHECK-NEXT: void foo3(queue_p &stream) {}
// CHECK-NEXT: void foo4(const queue_p &stream) {}
// CHECK-NEXT: void foo5(queue_p *stream) {}
// CHECK-NEXT: void foo6(queue_p &&stream) {}
#include <stdio.h>

void foo2(const cudaStream_t stream) {}
void foo3(cudaStream_t &stream) {}
void foo4(const cudaStream_t &stream) {}
void foo5(cudaStream_t *stream) {}
void foo6(cudaStream_t &&stream) {}

// CHECK: #define CS queue_p
#define CS cudaStream_t

// TODO: migrate types in typedef and using
typedef cudaStream_t CS2;
using CS3 = cudaStream_t;

void bar() {
  // CHECK: queue_p streams[10];
  cudaStream_t streams[10];
  CS s;
  CS2 s2;
  CS3 s3;
}
