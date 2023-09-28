// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/math/bfloat16/bfloat16_ext %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/math/bfloat16/bfloat16_ext/bfloat16_ext.dp.cpp

#include "cuda_bf16.h"

__global__ void kernelFuncBfloat162Arithmetic() {
  // CHECK: sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2;
  __nv_bfloat162 bf162, bf162_1, bf162_2;
  // CHECK: bf162 = bf162_1 / bf162_2;
  bf162 = __h2div(bf162_1, bf162_2);
}

__global__ void kernelFuncBfloat16Comparison() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16_1, bf16_2;
  __nv_bfloat16 bf16_1, bf16_2;
  bool b;
  // CHECK: b = bf16_1 == bf16_2;
  b = __heq(bf16_1, bf16_2);
}

int main() { return 0; }
