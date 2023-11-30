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
  // CHECK: b = sycl::isnan(float(bf16_1));
  b = __hisnan(bf16_1);
  // CHECK: b = bf16_1 <= bf16_2;
  b = __hle(bf16_1, bf16_2);
  // CHECK: b = dpct::unordered_compare(bf16_1, bf16_2, std::less_equal<>());
  b = __hleu(bf16_1, bf16_2);
  // CHECK: b = bf16_1 < bf16_2;
  b = __hlt(bf16_1, bf16_2);
  // CHECK: b = dpct::unordered_compare(bf16_1, bf16_2, std::less<>());
  b = __hltu(bf16_1, bf16_2);
  // CHECK: b = sycl::fmax(float(bf16_1), float(bf16_2));
  b = __hmax(bf16_1, bf16_2);
  // CHECK: b = dpct::fmax_nan(bf16_1, bf16_2);
  b = __hmax_nan(bf16_1, bf16_2);
  // CHECK: b = sycl::fmin(float(bf16_1), float(bf16_2));
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
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::isnan(float(bf162_1[0])), sycl::isnan(float(bf162_1[1])));
  bf162 = __hisnan2(bf162_1);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::less_equal<>());
  bf162 = __hle2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::less_equal<>());
  bf162 = __hleu2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::less<>());
  bf162 = __hlt2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::less<>());
  bf162 = __hltu2(bf162_1, bf162_2);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::fmax(float(bf162_1[0]), float(bf162_2[0])), sycl::fmax(float(bf162_1[1]), float(bf162_2[1])));
  bf162 = __hmax2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::fmax_nan(bf162_1, bf162_2);
  bf162 = __hmax2_nan(bf162_1, bf162_2);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::fmin(float(bf162_1[0]), float(bf162_2[0])), sycl::fmin(float(bf162_1[1]), float(bf162_2[1])));
  bf162 = __hmin2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::fmin_nan(bf162_1, bf162_2);
  bf162 = __hmin2_nan(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::compare(bf162_1, bf162_2, std::not_equal_to<>());
  bf162 = __hne2(bf162_1, bf162_2);
  // CHECK: bf162 = dpct::unordered_compare(bf162_1, bf162_2, std::not_equal_to<>());
  bf162 = __hneu2(bf162_1, bf162_2);
}

// CHECK: void test_conversions_device(sycl::ext::oneapi::bfloat16 *deviceArrayBFloat16) {
// CHECK-NEXT:   float f, f_1, f_2;
// CHECK-NEXT:   sycl::float2 f2, f2_1, f2_2;
// CHECK-NEXT:   sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2;
// CHECK-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2;
__global__ void test_conversions_device(__nv_bfloat16 *deviceArrayBFloat16) {
  float f, f_1, f_2;
  float2 f2, f2_1, f2_2;
  __nv_bfloat16 bf16, bf16_1, bf16_2;
  __nv_bfloat162 bf162, bf162_1, bf162_2;
  int i;
  long long ll;
  short s;
  unsigned u;
  unsigned long long ull;
  unsigned short us;
  double d;
  // CHECK: f2 = sycl::float2(bf162[0], bf162[1]);
  f2 = __bfloat1622float2(bf162);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf16, bf16);
  bf162 = __bfloat162bfloat162(bf16);
  // CHECK: f = static_cast<float>(bf16);
  f = __bfloat162float(bf16);
  // CHECK: i = sycl::vec<float, 1>(bf16).convert<int, sycl::rounding_mode::rtn>()[0];
  i = __bfloat162int_rd(bf16);
  // CHECK: i = sycl::vec<float, 1>(bf16).convert<int, sycl::rounding_mode::rte>()[0];
  i = __bfloat162int_rn(bf16);
  // CHECK: i = sycl::vec<float, 1>(bf16).convert<int, sycl::rounding_mode::rtp>()[0];
  i = __bfloat162int_ru(bf16);
  // CHECK: i = sycl::vec<float, 1>(bf16).convert<int, sycl::rounding_mode::rtz>()[0];
  i = __bfloat162int_rz(bf16);
  // CHECK: ll = sycl::vec<float, 1>(bf16).convert<long long, sycl::rounding_mode::rtn>()[0];
  ll = __bfloat162ll_rd(bf16);
  // CHECK: ll = sycl::vec<float, 1>(bf16).convert<long long, sycl::rounding_mode::rte>()[0];
  ll = __bfloat162ll_rn(bf16);
  // CHECK: ll = sycl::vec<float, 1>(bf16).convert<long long, sycl::rounding_mode::rtp>()[0];
  ll = __bfloat162ll_ru(bf16);
  // CHECK: ll = sycl::vec<float, 1>(bf16).convert<long long, sycl::rounding_mode::rtz>()[0];
  ll = __bfloat162ll_rz(bf16);
  // CHECK: s = sycl::vec<float, 1>(bf16).convert<short, sycl::rounding_mode::rtn>()[0];
  s = __bfloat162short_rd(bf16);
  // CHECK: s = sycl::vec<float, 1>(bf16).convert<short, sycl::rounding_mode::rte>()[0];
  s = __bfloat162short_rn(bf16);
  // CHECK: s = sycl::vec<float, 1>(bf16).convert<short, sycl::rounding_mode::rtp>()[0];
  s = __bfloat162short_ru(bf16);
  // CHECK: s = sycl::vec<float, 1>(bf16).convert<short, sycl::rounding_mode::rtz>()[0];
  s = __bfloat162short_rz(bf16);
  // CHECK: u = sycl::vec<float, 1>(bf16).convert<unsigned, sycl::rounding_mode::rtn>()[0];
  u = __bfloat162uint_rd(bf16);
  // CHECK: u = sycl::vec<float, 1>(bf16).convert<unsigned, sycl::rounding_mode::rte>()[0];
  u = __bfloat162uint_rn(bf16);
  // CHECK: u = sycl::vec<float, 1>(bf16).convert<unsigned, sycl::rounding_mode::rtp>()[0];
  u = __bfloat162uint_ru(bf16);
  // CHECK: u = sycl::vec<float, 1>(bf16).convert<unsigned, sycl::rounding_mode::rtz>()[0];
  u = __bfloat162uint_rz(bf16);
  // CHECK: ull = sycl::vec<float, 1>(bf16).convert<unsigned long long, sycl::rounding_mode::rtn>()[0];
  ull = __bfloat162ull_rd(bf16);
  // CHECK: ull = sycl::vec<float, 1>(bf16).convert<unsigned long long, sycl::rounding_mode::rte>()[0];
  ull = __bfloat162ull_rn(bf16);
  // CHECK: ull = sycl::vec<float, 1>(bf16).convert<unsigned long long, sycl::rounding_mode::rtp>()[0];
  ull = __bfloat162ull_ru(bf16);
  // CHECK: ull = sycl::vec<float, 1>(bf16).convert<unsigned long long, sycl::rounding_mode::rtz>()[0];
  ull = __bfloat162ull_rz(bf16);
  // CHECK: us = sycl::vec<float, 1>(bf16).convert<unsigned short, sycl::rounding_mode::rtn>()[0];
  us = __bfloat162ushort_rd(bf16);
  // CHECK: us = sycl::vec<float, 1>(bf16).convert<unsigned short, sycl::rounding_mode::rte>()[0];
  us = __bfloat162ushort_rn(bf16);
  // CHECK: us = sycl::vec<float, 1>(bf16).convert<unsigned short, sycl::rounding_mode::rtp>()[0];
  us = __bfloat162ushort_ru(bf16);
  // CHECK: us = sycl::vec<float, 1>(bf16).convert<unsigned short, sycl::rounding_mode::rtz>()[0];
  us = __bfloat162ushort_rz(bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __bfloat16_as_short is not supported.
  // CHECK-NEXT: */
  s = __bfloat16_as_short(bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __bfloat16_as_ushort is not supported.
  // CHECK-NEXT: */
  us = __bfloat16_as_ushort(bf16);
  // CHECK: bf16 = sycl::ext::oneapi::bfloat16(d);
  bf16 = __double2bfloat16(d);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f2[0], f2[1]);
  bf162 = __float22bfloat162_rn(f2);
  // CHECK: bf16 = sycl::ext::oneapi::bfloat16(f);
  bf16 = __float2bfloat16(f);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f, f);
  bf162 = __float2bfloat162_rn(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __float2bfloat16_rd is not supported.
  // CHECK-NEXT: */
  bf16 = __float2bfloat16_rd(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __float2bfloat16_rn is not supported.
  // CHECK-NEXT: */
  bf16 = __float2bfloat16_rn(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __float2bfloat16_ru is not supported.
  // CHECK-NEXT: */
  bf16 = __float2bfloat16_ru(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __float2bfloat16_rz is not supported.
  // CHECK-NEXT: */
  bf16 = __float2bfloat16_rz(f);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f_1, f_2);
  bf162 = __floats2bfloat162_rn(f_1, f_2);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf16_1, bf16_2);
  bf162 = __halves2bfloat162(bf16_1, bf16_2);
  // CHECK: bf16 = sycl::ext::oneapi::bfloat16(bf162[1]);
  bf16 = __high2bfloat16(bf162);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf162[1], bf162[1]);
  bf162 = __high2bfloat162(bf162);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf162_1[1], bf162_2[1]);
  bf162 = __highs2bfloat162(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __int2bfloat16_rd is not supported.
  // CHECK-NEXT: */
  bf16 = __int2bfloat16_rd(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __int2bfloat16_rn is not supported.
  // CHECK-NEXT: */
  bf16 = __int2bfloat16_rn(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __int2bfloat16_ru is not supported.
  // CHECK-NEXT: */
  bf16 = __int2bfloat16_ru(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __int2bfloat16_rz is not supported.
  // CHECK-NEXT: */
  bf16 = __int2bfloat16_rz(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ll2bfloat16_rd is not supported.
  // CHECK-NEXT: */
  bf16 = __ll2bfloat16_rd(ll);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ll2bfloat16_rn is not supported.
  // CHECK-NEXT: */
  bf16 = __ll2bfloat16_rn(ll);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ll2bfloat16_ru is not supported.
  // CHECK-NEXT: */
  bf16 = __ll2bfloat16_ru(ll);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ll2bfloat16_rz is not supported.
  // CHECK-NEXT: */
  bf16 = __ll2bfloat16_rz(ll);
  // CHECK: bf16 = sycl::ext::oneapi::bfloat16(bf162[0]);
  bf16 = __low2bfloat16(bf162);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf162[0], bf162[0]);
  bf162 = __low2bfloat162(bf162);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf162_1[0], bf162_2[0]);
  bf162 = __lows2bfloat162(bf162_1, bf162_2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __short2bfloat16_rd is not supported.
  // CHECK-NEXT: */
  bf16 = __short2bfloat16_rd(s);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __short2bfloat16_rn is not supported.
  // CHECK-NEXT: */
  bf16 = __short2bfloat16_rn(s);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __short2bfloat16_ru is not supported.
  // CHECK-NEXT: */
  bf16 = __short2bfloat16_ru(s);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __short2bfloat16_rz is not supported.
  // CHECK-NEXT: */
  bf16 = __short2bfloat16_rz(s);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __short_as_bfloat16 is not supported.
  // CHECK-NEXT: */
  bf16 = __short_as_bfloat16(s);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __uint2bfloat16_rd is not supported.
  // CHECK-NEXT: */
  bf16 = __uint2bfloat16_rd(u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __uint2bfloat16_rn is not supported.
  // CHECK-NEXT: */
  bf16 = __uint2bfloat16_rn(u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __uint2bfloat16_ru is not supported.
  // CHECK-NEXT: */
  bf16 = __uint2bfloat16_ru(u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __uint2bfloat16_rz is not supported.
  // CHECK-NEXT: */
  bf16 = __uint2bfloat16_rz(u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ull2bfloat16_rd is not supported.
  // CHECK-NEXT: */
  bf16 = __ull2bfloat16_rd(ull);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ull2bfloat16_rn is not supported.
  // CHECK-NEXT: */
  bf16 = __ull2bfloat16_rn(ull);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ull2bfloat16_ru is not supported.
  // CHECK-NEXT: */
  bf16 = __ull2bfloat16_ru(ull);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ull2bfloat16_rz is not supported.
  // CHECK-NEXT: */
  bf16 = __ull2bfloat16_rz(ull);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ushort2bfloat16_rd is not supported.
  // CHECK-NEXT: */
  bf16 = __ushort2bfloat16_rd(us);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ushort2bfloat16_rn is not supported.
  // CHECK-NEXT: */
  bf16 = __ushort2bfloat16_rn(us);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ushort2bfloat16_ru is not supported.
  // CHECK-NEXT: */
  bf16 = __ushort2bfloat16_ru(us);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ushort2bfloat16_rz is not supported.
  // CHECK-NEXT: */
  bf16 = __ushort2bfloat16_rz(us);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __ushort_as_bfloat16 is not supported.
  // CHECK-NEXT: */
  bf16 = __ushort_as_bfloat16(us);

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

__global__ void kernelFuncBfloat16Math() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16, bf16_1;
  __nv_bfloat16 bf16, bf16_1;
  // CHECK: bf16_1 = sycl::ceil(float(bf16));
  bf16_1 = hceil(bf16);
  // CHECK: bf16_1 = sycl::cos(float(bf16));
  bf16_1 = hcos(bf16);
  // CHECK: bf16_1 = sycl::exp(float(bf16));
  bf16_1 = hexp(bf16);
  // CHECK: bf16_1 = sycl::exp10(float(bf16));
  bf16_1 = hexp10(bf16);
  // CHECK: bf16_1 = sycl::exp2(float(bf16));
  bf16_1 = hexp2(bf16);
  // CHECK: bf16_1 = sycl::floor(float(bf16));
  bf16_1 = hfloor(bf16);
  // CHECK: bf16_1 = sycl::log(float(bf16));
  bf16_1 = hlog(bf16);
  // CHECK: bf16_1 = sycl::log10(float(bf16));
  bf16_1 = hlog10(bf16);
  // CHECK: bf16_1 = sycl::log2(float(bf16));
  bf16_1 = hlog2(bf16);
  // CHECK: bf16_1 = sycl::half_precision::recip(float(bf16));
  bf16_1 = hrcp(bf16);
  // CHECK: bf16_1 = sycl::rint(float(bf16));
  bf16_1 = hrint(bf16);
  // CHECK: bf16_1 = sycl::rsqrt(float(bf16));
  bf16_1 = hrsqrt(bf16);
  // CHECK: bf16_1 = sycl::sin(float(bf16));
  bf16_1 = hsin(bf16);
  // CHECK: bf16_1 = sycl::sqrt(float(bf16));
  bf16_1 = hsqrt(bf16);
  // CHECK: bf16_1 = sycl::trunc(float(bf16));
  bf16_1 = htrunc(bf16);
}

__global__ void kernelFuncBfloat162Math() {
  // CHECK: sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1;
  __nv_bfloat162 bf162, bf162_1;
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::ceil(float(bf162[0])), sycl::ceil(float(bf162[1])));
  bf162_1 = h2ceil(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::cos(float(bf162[0])), sycl::cos(float(bf162[1])));
  bf162_1 = h2cos(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::exp(float(bf162[0])), sycl::exp(float(bf162[1])));
  bf162_1 = h2exp(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::exp10(float(bf162[0])), sycl::exp10(float(bf162[1])));
  bf162_1 = h2exp10(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::exp2(float(bf162[0])), sycl::exp2(float(bf162[1])));
  bf162_1 = h2exp2(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::floor(float(bf162[0])), sycl::floor(float(bf162[1])));
  bf162_1 = h2floor(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::log(float(bf162[0])), sycl::log(float(bf162[1])));
  bf162_1 = h2log(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::log10(float(bf162[0])), sycl::log10(float(bf162[1])));
  bf162_1 = h2log10(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::log2(float(bf162[0])), sycl::log2(float(bf162[1])));
  bf162_1 = h2log2(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::half_precision::recip(float(bf162[0])), sycl::half_precision::recip(float(bf162[1])));
  bf162_1 = h2rcp(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::rint(float(bf162[0])), sycl::rint(float(bf162[1])));
  bf162_1 = h2rint(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::rsqrt(float(bf162[0])), sycl::rsqrt(float(bf162[1])));
  bf162_1 = h2rsqrt(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::sin(float(bf162[0])), sycl::sin(float(bf162[1])));
  bf162_1 = h2sin(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::sqrt(float(bf162[0])), sycl::sqrt(float(bf162[1])));
  bf162_1 = h2sqrt(bf162);
  // CHECK: bf162_1 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::trunc(float(bf162[0])), sycl::trunc(float(bf162[1])));
  bf162_1 = h2trunc(bf162);
}

int main() { return 0; }
