// RUN: dpct --format-range=none -out-root %T/math/cuda-math-need-paren %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-need-paren/cuda-math-need-paren.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

using namespace std;

void __global__ kernel() {
  half2 h2;
  // CHECK: (h2 + h2).convert<float, sycl::rounding_mode::automatic>();
  __half22float2(__hadd2(h2, h2));
}

int main() { return 0; }
