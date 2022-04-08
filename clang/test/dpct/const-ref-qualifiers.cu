// RUN: c2s --usm-level=none -out-root %T/const-ref-qualifiers %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/const-ref-qualifiers/const-ref-qualifiers.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <stdio.h>
// CHECK-EMPTY:
// CHECK-NEXT: void foo2(sycl::queue *const stream) {}
// CHECK-NEXT: void foo3(sycl::queue *&stream) {}
// CHECK-NEXT: void foo4(sycl::queue *const &stream) {}
// CHECK-NEXT: void foo5(sycl::queue **stream) {}
// CHECK-NEXT: void foo6(sycl::queue *&&stream) {}
#include <stdio.h>

void foo2(const cudaStream_t stream) {}
void foo3(cudaStream_t &stream) {}
void foo4(const cudaStream_t &stream) {}
void foo5(cudaStream_t *stream) {}
void foo6(cudaStream_t &&stream) {}

// CHECK: #define CS sycl::queue *
#define CS cudaStream_t

// TODO: migrate types in typedef and using
typedef cudaStream_t CS2;
using CS3 = cudaStream_t;

void bar() {
  // CHECK: sycl::queue *streams[10];
  cudaStream_t streams[10];
  CS s;
  CS2 s2;
  CS3 s3;
}

