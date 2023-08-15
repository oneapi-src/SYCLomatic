// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/math/cuda-math-intrinsics-cuda8.0-not-support %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-intrinsics-cuda8.0-not-support/cuda-math-intrinsics-cuda8.0-not-support.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncHalf() {
  __half h, h_1, h_2;

  // Half Arithmetic Functions

  // CHECK: h_2 = h / h_1;
  h_2 = __hdiv(h, h_1);
}

__global__ void kernelFuncHalf2() {
  __half2 h2, h2_1, h2_2;

  // Half2 Arithmetic Functions

  // CHECK: h2_2 = h2 / h2_1;
  h2_2 = __h2div(h2, h2_1);
}

int main() { return 0; }
