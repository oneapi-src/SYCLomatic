// REQUIRES: cuda-8.0
// REQUIRES: v8.0
// RUN: dpct --format-range=none -out-root %T/cuda-math-intrinsics1 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda-math-intrinsics1/cuda-math-intrinsics1.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncHalf() {
  __half h, h_1, h_2;
  bool b;

  // Half Arithmetic Functions

  // CHECK: h_2 = h / h_1;
  h_2 = hdiv(h, h_1);

}

__global__ void kernelFuncHalf2() {
  __half2 h2, h2_1, h2_2;
  bool b;
  // CHECK: h2 = h2_1 / h2_2;
  h2 = h2div(h2_1, h2_2);
}
int main() { return 0; }
