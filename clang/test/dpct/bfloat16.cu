// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/bfloat16 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/bfloat16/bfloat16.dp.cpp

// CHECK: #include <oneapi/mkl/bfloat16.hpp>
#include "cuda_bf16.h"

// CHECK: void foo(oneapi::mkl::bfloat16 *a) {
void foo(__nv_bfloat16 *a) {
  int i = 0;
  float f = 3.0f;
// CHECK: a[i] = (oneapi::mkl::bfloat16)f;
  a[i] = (__nv_bfloat16)f;
}

void test_conversions() {
  // CHECK: const auto bf16 = oneapi::mkl::bfloat16(3.14f);
  const auto bf16 = __float2bfloat16(3.14f);

  // CHECK: const float f32 = static_cast<float>(bf16);
  const float f32 = __bfloat162float(bf16);
}
