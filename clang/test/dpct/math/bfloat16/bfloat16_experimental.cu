// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none --use-experimental-features=bfloat16_math_functions -out-root %T/math/bfloat16/bfloat16_experimental %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/math/bfloat16/bfloat16_experimental/bfloat16_experimental.dp.cpp

#include "cuda_bf16.h"

__global__ void kernelFuncBfloat16Arithmetic() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2, bf16_3;
  __nv_bfloat16 bf16, bf16_1, bf16_2, bf16_3;
  // CHECK: bf16 = sycl::ext::oneapi::experimental::fabs(bf16_1);
  bf16 = __habs(bf16_1);
  // CHECK: bf16 = bf16_1 + bf16_2;
  bf16 = __hadd(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 + bf16_2;
  bf16 = __hadd_rn(bf16_1, bf16_2);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(bf16_1 + bf16_2, 0.f, 1.0f);
  bf16 = __hadd_sat(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 / bf16_2;
  bf16 = __hdiv(bf16_1, bf16_2);
  // CHECK: bf16 = sycl::ext::oneapi::experimental::fma(bf16_1, bf16_2, bf16_3);
  bf16 = __hfma(bf16_1, bf16_2, bf16_3);
  // CHECK: bf16 = dpct::relu(sycl::ext::oneapi::experimental::fma(bf16_1, bf16_2, bf16_3));
  bf16 = __hfma_relu(bf16_1, bf16_2, bf16_3);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::experimental::fma(bf16_1, bf16_2, bf16_3), 0.f, 1.0f);
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
  // CHECK: bf162 = sycl::ext::oneapi::experimental::fabs(bf162_1);
  bf162 = __habs2(bf162_1);
  // CHECK: bf162 = bf162_1 + bf162_2;
  bf162 = __hadd2(bf162_1, bf162_2);
  // CHECK: bf162 = bf162_1 + bf162_2;
  bf162 = __hadd2_rn(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::clamp(bf162_1 + bf162_2, {0.f, 0.f}, {1.f, 1.f});
  bf162 = __hadd2_sat(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::complex_mul_add(bf162_1, bf162_2, bf162_3);
  bf162 = __hcmadd(bf162_1, bf162_2, bf162_3);
  // CHECK: bf162 = sycl::ext::oneapi::experimental::fma(bf162_1, bf162_2, bf162_3);
  bf162 = __hfma2(bf162_1, bf162_2, bf162_3);
  // CHECK: bf162 = dpct::relu(sycl::ext::oneapi::experimental::fma(bf162_1, bf162_2, bf162_3));
  bf162 = __hfma2_relu(bf162_1, bf162_2, bf162_3);
  // CHECK: bf162 = dpct::clamp(sycl::ext::oneapi::experimental::fma(bf162_1, bf162_2, bf162_3), {0.f, 0.f}, {1.f, 1.f});
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

int main() { return 0; }
