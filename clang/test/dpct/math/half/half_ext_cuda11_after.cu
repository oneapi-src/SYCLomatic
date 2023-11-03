// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/math/half/half_ext_cuda11_after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/math/half/half_ext_cuda11_after/half_ext_cuda11_after.dp.cpp

#include "cuda_fp16.h"

__global__ void kernelFuncHalfConversion() {
  half h;
  half2 h2;
  // CHECK: h2 = sycl::half2(h, h);
  h2 = make_half2(h, h);
}

int main() {
  kernelFuncHalfConversion<<<1, 1>>>();
  return 0;
}
