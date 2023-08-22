// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/math/cuda-math-extension-cuda9-after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-extension-cuda9-after/cuda-math-extension-cuda9-after.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncHalf() {
  __half h, h_1, h_2;
  bool b;

  // Half Arithmetic Functions

  // CHECK: h_2 = sycl::ext::intel::math::hdiv(h, h_1);
  h_2 = __hdiv(h, h_1);
}

int main() { return 0; }
