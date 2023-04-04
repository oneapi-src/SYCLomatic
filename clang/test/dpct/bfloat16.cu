// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/bfloat16 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/bfloat16/bfloat16.dp.cpp

#include "cuda_bf16.h"

// CHECK: void foo(sycl::ext::oneapi::bfloat16 *a) {
void foo(__nv_bfloat16 *a) {
  int i = 0;
  float f = 3.0f;
  // CHECK: a[i] = (sycl::ext::oneapi::bfloat16)f;
  a[i] = (__nv_bfloat16)f;
}

// CHECK: void test_conversions_device() {
// CHECK-NEXT:   float f, f_1, f_2;
// CHECK-NEXT:   sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2;
// CHECK-NEXT:   f = static_cast<float>(bf16);
// CHECK-NEXT:   bf16 = sycl::ext::oneapi::bfloat16(f);
__global__ void test_conversions_device() {
  float f, f_1, f_2;
  __nv_bfloat16 bf16, bf16_1, bf16_2;
  f = __bfloat162float(bf16);
  bf16 = __float2bfloat16(f);
}

// CHECK: void test_conversions() {
// CHECK-NEXT:   float f, f_1, f_2;
// CHECK-NEXT:   sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2;
// CHECK-NEXT:   f = static_cast<float>(bf16);
// CHECK-NEXT:   bf16 = sycl::ext::oneapi::bfloat16(f);
void test_conversions() {
  float f, f_1, f_2;
  __nv_bfloat16 bf16, bf16_1, bf16_2;
  f = __bfloat162float(bf16);
  bf16 = __float2bfloat16(f);
}

int main() { return 0; }
