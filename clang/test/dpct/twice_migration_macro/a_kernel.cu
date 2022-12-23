// RUN: echo

#ifndef __A_KERNEL_CU__
#define __A_KERNEL_CU__

#include "a.h"
#include "cuda.h"

// CHECK: void kernel_1() {
// CHECK-NEXT:   sycl::log2((float)MACRO_A);
// CHECK-NEXT:   int xyz = MACRO_A;
// CHECK-NEXT: }
__global__ void kernel_1() {
  __log2f(MACRO_A);
  int xyz = MACRO_A;
}

// CHECK: void kernel_2() {}
__global__ void kernel_2() {}
#endif
