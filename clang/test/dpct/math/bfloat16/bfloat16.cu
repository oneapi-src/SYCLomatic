// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none -out-root %T/math/bfloat16/bfloat16 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/math/bfloat16/bfloat16/bfloat16.dp.cpp

#include "cuda_bf16.h"

// CHECK: class C : public sycl::marray<sycl::ext::oneapi::bfloat16, 2> {
class C : public __nv_bfloat162 {
  void f() {
    // CHECK: (*this)[0];
    // CHECK-NEXT: (*this)[1];
    x;
    y;
  }
};

// CHECK: void foo(sycl::ext::oneapi::bfloat16 *a, sycl::marray<sycl::ext::oneapi::bfloat16, 2> *b) {
void foo(__nv_bfloat16 *a, __nv_bfloat162 *b) {
  int i = 0;
  float f = 3.0f;
  // CHECK: a[i] = (sycl::ext::oneapi::bfloat16)f;
  a[i] = (__nv_bfloat16)f;

  // CHECK: (*b)[0];
  // CHECK-NEXT: (*b)[1];
  b->x;
  b->y;
}

__global__ void kernelFuncBfloat16Arithmetic() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2, bf16_3;
  __nv_bfloat16 bf16, bf16_1, bf16_2, bf16_3;
  // CHECK: bf16 = sycl::fabs(float(bf16_1));
  bf16 = __habs(bf16_1);
  // CHECK: bf16 = bf16_1 + bf16_2;
  bf16 = __hadd(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 + bf16_2;
  bf16 = __hadd_rn(bf16_1, bf16_2);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(bf16_1 + bf16_2, 0.f, 1.0f);
  bf16 = __hadd_sat(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 / bf16_2;
  bf16 = __hdiv(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 * bf16_2 + bf16_3;
  bf16 = __hfma(bf16_1, bf16_2, bf16_3);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hfma_relu is not supported.
  // CHECK-NEXT: */
  bf16 = __hfma_relu(bf16_1, bf16_2, bf16_3);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(bf16_1 * bf16_2 + bf16_3, 0.f, 1.0f);
  bf16 = __hfma_sat(bf16_1, bf16_2, bf16_3);
  // CHECK: bf16 = bf16_1 * bf16_2;
  bf16 = __hmul(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 * bf16_2;
  bf16 = __hmul_rn(bf16_1, bf16_2);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(bf16_1 * bf16_2, 0.f, 1.0f);
  bf16 = __hmul_sat(bf16_1, bf16_2);
  // CHECK: bf16 = -bf16_1;
  bf16 = __hneg(bf16_1);
  // CHECK: bf16 = bf16_1 - bf16_2;
  bf16 = __hsub(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 - bf16_2;
  bf16 = __hsub_rn(bf16_1, bf16_2);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(bf16_1 - bf16_2, 0.f, 1.0f);
  bf16 = __hsub_sat(bf16_1, bf16_2);
}

__global__ void kernelFuncBfloat162Arithmetic() {
  // CHECK: sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2, bf162_3;
  __nv_bfloat162 bf162, bf162_1, bf162_2, bf162_3;
  // CHECK: bf162 = bf162_1 / bf162_2;
  bf162 = __h2div(bf162_1, bf162_2);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::fabs(float(bf162_1[0])), sycl::fabs(float(bf162_1[1])));
  bf162 = __habs2(bf162_1);
  // CHECK: bf162 = bf162_1 + bf162_2;
  bf162 = __hadd2(bf162_1, bf162_2);
  // CHECK: bf162 = bf162_1 + bf162_2;
  bf162 = __hadd2_rn(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::clamp(bf162_1 + bf162_2, {0.f, 0.f}, {1.f, 1.f});
  bf162 = __hadd2_sat(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::complex_mul_add(bf162_1, bf162_2, bf162_3);
  bf162 = __hcmadd(bf162_1, bf162_2, bf162_3);
  // CHECK: bf162 = bf162_1 * bf162_2 + bf162_3;
  bf162 = __hfma2(bf162_1, bf162_2, bf162_3);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hfma2_relu is not supported.
  // CHECK-NEXT: */
  bf162 = __hfma2_relu(bf162_1, bf162_2, bf162_3);
  // CHECK: bf162 = dpct::clamp(bf162_1 * bf162_2 + bf162_3, {0.f, 0.f}, {1.f, 1.f});
  bf162 = __hfma2_sat(bf162_1, bf162_2, bf162_3);
  // CHECK: bf162 = bf162_1 * bf162_2;
  bf162 = __hmul2(bf162_1, bf162_2);
  // CHECK: bf162 = bf162_1 * bf162_2;
  bf162 = __hmul2_rn(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::clamp(bf162_1 * bf162_2, {0.f, 0.f}, {1.f, 1.f});
  bf162 = __hmul2_sat(bf162_1, bf162_2);
  // CHECK: bf162 = -bf162_1;
  bf162 = __hneg2(bf162_1);
  // CHECK: bf162 = bf162_1 - bf162_2;
  bf162 = __hsub2(bf162_1, bf162_2);
  // CHECK: bf162 = bf162_1 - bf162_2;
  bf162 = __hsub2_rn(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::clamp(bf162_1 - bf162_2, {0.f, 0.f}, {1.f, 1.f});
  bf162 = __hsub2_sat(bf162_1, bf162_2);
}

// CHECK: void test_conversions_device(sycl::ext::oneapi::bfloat16 *deviceArrayBFloat16) {
// CHECK-NEXT:   float f, f_1, f_2;
// CHECK-NEXT:   sycl::float2 f2, f2_1, f2_2;
// CHECK-NEXT:   sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2;
// CHECK-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2;
// CHECK-NEXT:   f2 = sycl::float2(bf162[0], bf162[1]);
// CHECK-NEXT:   f = static_cast<float>(bf16);
// CHECK-NEXT:   bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f2[0], f2[1]);
// CHECK-NEXT:   bf16 = sycl::ext::oneapi::bfloat16(f);
__global__ void test_conversions_device(__nv_bfloat16 *deviceArrayBFloat16) {
  float f, f_1, f_2;
  float2 f2, f2_1, f2_2;
  __nv_bfloat16 bf16, bf16_1, bf16_2;
  __nv_bfloat162 bf162, bf162_1, bf162_2;
  f2 = __bfloat1622float2(bf162);
  f = __bfloat162float(bf16);
  bf162 = __float22bfloat162_rn(f2);
  bf16 = __float2bfloat16(f);

  // CHECK:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = *deviceArrayBFloat16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = bf16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf162_2 = bf162;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = *deviceArrayBFloat16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = bf16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf162_2 = bf162;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = *deviceArrayBFloat16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = bf16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf162_2 = bf162;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = *deviceArrayBFloat16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = bf16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf162_2 = bf162;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = *deviceArrayBFloat16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = bf16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf162_2 = bf162;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = *deviceArrayBFloat16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf16_2 = bf16;
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   bf162_2 = bf162;
  bf16_2 = __ldca(deviceArrayBFloat16);
  bf16_2 = __ldca(&bf16);
  bf162_2 = __ldca(&bf162);
  bf16_2 = __ldcg(deviceArrayBFloat16);
  bf16_2 = __ldcg(&bf16);
  bf162_2 = __ldcg(&bf162);
  bf16_2 = __ldcs(deviceArrayBFloat16);
  bf16_2 = __ldcs(&bf16);
  bf162_2 = __ldcs(&bf162);
  bf16_2 = __ldcv(deviceArrayBFloat16);
  bf16_2 = __ldcv(&bf16);
  bf162_2 = __ldcv(&bf162);
  bf16_2 = __ldg(deviceArrayBFloat16);
  bf16_2 = __ldg(&bf16);
  bf162_2 = __ldg(&bf162);
  bf16_2 = __ldlu(deviceArrayBFloat16);
  bf16_2 = __ldlu(&bf16);
  bf162_2 = __ldlu(&bf162);

  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(deviceArrayBFloat16 + 1) = bf16;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: bf16_2 = bf16;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: bf162_2 = bf162;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *deviceArrayBFloat16 = bf16;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: bf16_2 = bf16;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: bf162_2 = bf162;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(deviceArrayBFloat16 + 1) = bf16;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: bf16_2 = bf16;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: bf162_2 = bf162;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *deviceArrayBFloat16 = bf16;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: bf16_2 = bf16;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: bf162_2 = bf162;
  __stcg(deviceArrayBFloat16 + 1, bf16);
  __stcg(&bf16_2, bf16);
  __stcg(&bf162_2, bf162);
  __stcs(deviceArrayBFloat16, bf16);
  __stcs(&bf16_2, bf16);
  __stcs(&bf162_2, bf162);
  __stwb(deviceArrayBFloat16 + 1, bf16);
  __stwb(&bf16_2, bf16);
  __stwb(&bf162_2, bf162);
  __stwt(deviceArrayBFloat16, bf16);
  __stwt(&bf16_2, bf16);
  __stwt(&bf162_2, bf162);
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
