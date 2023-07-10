// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none --use-experimental-features=bfloat16 -out-root %T/bfloat16_experimental %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/bfloat16_experimental/bfloat16_experimental.dp.cpp

#include "cuda_bf16.h"

__global__ void kernelFuncBfloat16Arithmetic() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16, bf16_1;
  __nv_bfloat16 bf16, bf16_1;
  // CHECK: bf16 = sycl::ext::oneapi::experimental::fabs(bf16_1);
  bf16 = __habs(bf16_1);
}

int main() { return 0; }
