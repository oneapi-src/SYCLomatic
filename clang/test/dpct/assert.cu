// RUN: dpct --format-range=none -out-root %T/assert %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/assert/assert.dp.cpp
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>

__global__ void kernel_assert(int *d_ptr, int length) {
  // CHECK: assert(0);
  // CHECK-NEXT: assert(0);
  // CHECK-NEXT: assert(d_ptr);
  __assert_fail("", "", 1, "");  
  __assertfail("", "", 1, "", sizeof(char));
  assert(d_ptr);
}

__device__ void device_assert(int *d_ptr, int length) {
  // CHECK: assert(0);
  // CHECK-NEXT: assert(0);
  // CHECK-NEXT: assert(d_ptr);
  __assert_fail("", "", 1, "");  
  __assertfail("", "", 1, "", sizeof(char));
  assert(d_ptr);
}

