// RUN: dpct --usm-level=none -out-root %T/insert-header-using %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/insert-header-using/insert-header-using.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdio.h>
// CHECK-EMPTY:
// CHECK-NEXT: #ifndef _TEST_
// CHECK-NEXT: #define _TEST_
#include <stdio.h>

#ifndef _TEST_
#define _TEST_

__global__ void hello() {
  // CHECK: sycl::queue *stream;
  cudaStream_t stream;
}

#endif

