// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/movmatrix %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/movmatrix/movmatrix.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/movmatrix/movmatrix.dp.cpp -o %T/movmatrix/movmatrix.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>

using bf16_2 = __nv_bfloat162;

//Syntax:
//movmatrix.sync.aligned.shape.trans.type d, a;
//.shape = {.m8n8};
//.type = {.b16};#include <cuda_bf16.h>
// Only .m8n8.b16
// 

__global__ void movmatrix(bf16_2 &dst, const bf16_2 &src) {

  // CHECK: dpct::experimental::matrix::movmatrix(*(uint32_t *)(&dst), (*(uint32_t *)(&src)));
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
               : "+r"(*(uint32_t *)(&dst))
               : "r"(*(uint32_t *)(&src)));
}

// clang-format on