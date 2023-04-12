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
// CHECK-NEXT:   sycl::float2 f2, f2_1, f2_2;
// CHECK-NEXT:   sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2;
// CHECK-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2;
// CHECK-NEXT:   f2 = sycl::float2(bf162[0], bf162[1]);
// CHECK-NEXT:   f = static_cast<float>(bf16);
// CHECK-NEXT:   bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f2[0], f2[1]);
// CHECK-NEXT:   bf16 = sycl::ext::oneapi::bfloat16(f);
__global__ void test_conversions_device() {
  float f, f_1, f_2;
  float2 f2, f2_1, f2_2;
  __nv_bfloat16 bf16, bf16_1, bf16_2;
  __nv_bfloat162 bf162, bf162_1, bf162_2;
  f2 = __bfloat1622float2(bf162);
  f = __bfloat162float(bf16);
  bf162 = __float22bfloat162_rn(f2);
  bf16 = __float2bfloat16(f);
}

// CHECK: void test_conversions() {
// CHECK-NEXT:   float f, f_1, f_2;
// CHECK-NEXT:   sycl::float2 f2, f2_1, f2_2;
// CHECK-NEXT:   sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2;
// CHECK-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2;
// CHECK-NEXT:   f2 = sycl::float2(bf162[0], bf162[1]);
// CHECK-NEXT:   f = static_cast<float>(bf16);
// CHECK-NEXT:   bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f2[0], f2[1]);
// CHECK-NEXT:   bf16 = sycl::ext::oneapi::bfloat16(f);
void test_conversions() {
  float f, f_1, f_2;
  float2 f2, f2_1, f2_2;
  __nv_bfloat16 bf16, bf16_1, bf16_2;
  __nv_bfloat162 bf162, bf162_1, bf162_2;
  f2 = __bfloat1622float2(bf162);
  f = __bfloat162float(bf16);
  bf162 = __float22bfloat162_rn(f2);
  bf16 = __float2bfloat16(f);
}

int main() { return 0; }
