// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/cuda-math-half2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda-math-half2/cuda-math-half2.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <cuda_fp16.h>

__global__ void test() {
  __half2 h2, h2_2;

  // Half2 Accessors

  // CHECK: h2.x();
  h2.x;
  // CHECK: h2.y();
  h2.y;
}
