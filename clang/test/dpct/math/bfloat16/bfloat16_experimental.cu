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

__global__ void kernelFuncBfloat16Comparison() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16_1, bf16_2;
  __nv_bfloat16 bf16_1, bf16_2;
  bool b;
  // CHECK: b = bf16_1 == bf16_2;
  b = __heq(bf16_1, bf16_2);
  // CHECK: b = dpct::unordered_compare(bf16_1, bf16_2, std::equal_to<>());
  b = __hequ(bf16_1, bf16_2);
  // CHECK: b = bf16_1 >= bf16_2;
  b = __hge(bf16_1, bf16_2);
  // CHECK: b = dpct::unordered_compare(bf16_1, bf16_2, std::greater_equal<>());
  b = __hgeu(bf16_1, bf16_2);
  // CHECK: b = bf16_1 > bf16_2;
  b = __hgt(bf16_1, bf16_2);
  // CHECK: b = dpct::unordered_compare(bf16_1, bf16_2, std::greater<>());
  b = __hgtu(bf16_1, bf16_2);
  // CHECK: b = sycl::isinf(float(bf16_1));
  b = __hisinf(bf16_1);
  // CHECK: b = sycl::ext::oneapi::experimental::isnan(bf16_1);
  b = __hisnan(bf16_1);
  // CHECK: b = bf16_1 <= bf16_2;
  b = __hle(bf16_1, bf16_2);
  // CHECK: b = dpct::unordered_compare(bf16_1, bf16_2, std::less_equal<>());
  b = __hleu(bf16_1, bf16_2);
  // CHECK: b = bf16_1 < bf16_2;
  b = __hlt(bf16_1, bf16_2);
  // CHECK: b = dpct::unordered_compare(bf16_1, bf16_2, std::less<>());
  b = __hltu(bf16_1, bf16_2);
  // CHECK: b = sycl::ext::oneapi::experimental::fmax(bf16_1, bf16_2);
  b = __hmax(bf16_1, bf16_2);
  // CHECK: b = dpct::fmax_nan(bf16_1, bf16_2);
  b = __hmax_nan(bf16_1, bf16_2);
  // CHECK: b = sycl::ext::oneapi::experimental::fmin(bf16_1, bf16_2);
  b = __hmin(bf16_1, bf16_2);
  // CHECK: b = dpct::fmin_nan(bf16_1, bf16_2);
  b = __hmin_nan(bf16_1, bf16_2);
  // CHECK: b = dpct::compare(bf16_1, bf16_2, std::not_equal_to<>());
  b = __hne(bf16_1, bf16_2);
  // CHECK: b = dpct::unordered_compare(bf16_1, bf16_2, std::not_equal_to<>());
  b = __hneu(bf16_1, bf16_2);
}

__global__ void kernelFuncBfloat162Comparison() {
  // CHECK: sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2;
  __nv_bfloat162 bf162, bf162_1, bf162_2;
  bool b;
  // CHECK: b = dpct::compare_both(bf162_1, bf162_2, std::equal_to<>());
  b = __hbeq2(bf162_1, bf162_2);
  // CHECK: b = dpct::unordered_compare_both(bf162_1, bf162_2, std::equal_to<>());
  b = __hbequ2(bf162_1, bf162_2);
  // CHECK: b = dpct::compare_both(bf162_1, bf162_2, std::greater_equal<>());
  b = __hbge2(bf162_1, bf162_2);
  // CHECK: b = dpct::unordered_compare_both(bf162_1, bf162_2, std::greater_equal<>());
  b = __hbgeu2(bf162_1, bf162_2);
  // CHECK: b = dpct::compare_both(bf162_1, bf162_2, std::greater<>());
  b = __hbgt2(bf162_1, bf162_2);
  // CHECK: b = dpct::unordered_compare_both(bf162_1, bf162_2, std::greater<>());
  b = __hbgtu2(bf162_1, bf162_2);
  // CHECK: b = dpct::compare_both(bf162_1, bf162_2, std::less_equal<>());
  b = __hble2(bf162_1, bf162_2);
  // CHECK: b = dpct::unordered_compare_both(bf162_1, bf162_2, std::less_equal<>());
  b = __hbleu2(bf162_1, bf162_2);
  // CHECK: b = dpct::compare_both(bf162_1, bf162_2, std::less<>());
  b = __hblt2(bf162_1, bf162_2);
  // CHECK: b = dpct::unordered_compare_both(bf162_1, bf162_2, std::less<>());
  b = __hbltu2(bf162_1, bf162_2);
  // CHECK: b = dpct::compare_both(bf162_1, bf162_2, std::not_equal_to<>());
  b = __hbne2(bf162_1, bf162_2);
  // CHECK: b = dpct::unordered_compare_both(bf162_1, bf162_2, std::not_equal_to<>());
  b = __hbneu2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::equal_to<>());
  bf162 = __heq2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::equal_to<>());
  bf162 = __hequ2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::greater_equal<>());
  bf162 = __hge2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::greater_equal<>());
  bf162 = __hgeu2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::greater<>());
  bf162 = __hgt2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::greater<>());
  bf162 = __hgtu2(bf162_1, bf162_2);
  // CHECK: bf162 = sycl::ext::oneapi::experimental::isnan(bf162_1);
  bf162 = __hisnan2(bf162_1);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::less_equal<>());
  bf162 = __hle2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::less_equal<>());
  bf162 = __hleu2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::less<>());
  bf162 = __hlt2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::less<>());
  bf162 = __hltu2(bf162_1, bf162_2);
  // CHECK: bf162 = sycl::ext::oneapi::experimental::fmax(bf162_1, bf162_2);
  bf162 = __hmax2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::fmax_nan(bf162_1, bf162_2);
  bf162 = __hmax2_nan(bf162_1, bf162_2);
  // CHECK: bf162 = sycl::ext::oneapi::experimental::fmin(bf162_1, bf162_2);
  bf162 = __hmin2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::fmin_nan(bf162_1, bf162_2);
  bf162 = __hmin2_nan(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::not_equal_to<>());
  bf162 = __hne2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::not_equal_to<>());
  bf162 = __hneu2(bf162_1, bf162_2);
}

__global__ void kernelFuncBfloat16Math() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16, bf16_1;
  __nv_bfloat16 bf16, bf16_1;
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::ceil(bf16);
  bf16_1 = hceil(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::cos(bf16);
  bf16_1 = hcos(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::exp(bf16);
  bf16_1 = hexp(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::exp10(bf16);
  bf16_1 = hexp10(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::exp2(bf16);
  bf16_1 = hexp2(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::floor(bf16);
  bf16_1 = hfloor(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::log(bf16);
  bf16_1 = hlog(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::log10(bf16);
  bf16_1 = hlog10(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::log2(bf16);
  bf16_1 = hlog2(bf16);
  // CHECK: bf16_1 = sycl::half_precision::recip(float(bf16));
  bf16_1 = hrcp(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::rint(bf16);
  bf16_1 = hrint(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::rsqrt(bf16);
  bf16_1 = hrsqrt(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::sin(bf16);
  bf16_1 = hsin(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::sqrt(bf16);
  bf16_1 = hsqrt(bf16);
  // CHECK: bf16_1 = sycl::ext::oneapi::experimental::trunc(bf16);
  bf16_1 = htrunc(bf16);
}

__global__ void kernelFuncBfloat162Math() {
  // CHECK: sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1;
  __nv_bfloat162 bf162, bf162_1;
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::ceil(bf162);
  bf162_1 = h2ceil(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::cos(bf162);
  bf162_1 = h2cos(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::exp(bf162);
  bf162_1 = h2exp(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::exp10(bf162);
  bf162_1 = h2exp10(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::exp2(bf162);
  bf162_1 = h2exp2(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::floor(bf162);
  bf162_1 = h2floor(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::log(bf162);
  bf162_1 = h2log(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::log10(bf162);
  bf162_1 = h2log10(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::log2(bf162);
  bf162_1 = h2log2(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::half_precision::recip(float(bf162[0])), sycl::half_precision::recip(float(bf162[1])));
  bf162_1 = h2rcp(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::rint(bf162);
  bf162_1 = h2rint(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::rsqrt(bf162);
  bf162_1 = h2rsqrt(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::sin(bf162);
  bf162_1 = h2sin(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::sqrt(bf162);
  bf162_1 = h2sqrt(bf162);
  // CHECK: bf162_1 = sycl::ext::oneapi::experimental::trunc(bf162);
  bf162_1 = h2trunc(bf162);
}

int main() { return 0; }
