// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/math/cuda-math-syclcompat %s -use-syclcompat --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-syclcompat/cuda-math-syclcompat.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST %T/math/cuda-math-syclcompat/cuda-math-syclcompat.dp.cpp -o %T/math/cuda-math-syclcompat/cuda-math-syclcompat.dp.o %}

#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void kernelFuncBfloat162Arithmetic() {
  __nv_bfloat162 bf162, bf162_1, bf162_2, bf162_3;
  // CHECK: bf162 = syclcompat::cmul_add(bf162_1, bf162_2, bf162_3);
  bf162 = __hcmadd(bf162_1, bf162_2, bf162_3);

  __half2 h2, h2_1, h2_2;
  // CHECK: h2_2 = syclcompat::cmul_add(h2, h2_1, h2_2);
  h2_2 = __hcmadd(h2, h2_1, h2_2);
}