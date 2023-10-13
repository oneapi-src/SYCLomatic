// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/math/bfloat16/bfloat16_ext %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/math/bfloat16/bfloat16_ext/bfloat16_ext.dp.cpp

#include "cuda_bf16.h"

__global__ void kernelFuncBfloat162Arithmetic() {
  // CHECK: sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2;
  __nv_bfloat162 bf162, bf162_1, bf162_2;
  // CHECK: bf162 = bf162_1 / bf162_2;
  bf162 = __h2div(bf162_1, bf162_2);
}

__global__ void kernelFuncBfloat16Comparison() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16_1, bf16_2;
  __nv_bfloat16 bf16_1, bf16_2;
  bool b;
  // CHECK: b = bf16_1 == bf16_2;
  b = __heq(bf16_1, bf16_2);
}

__global__ void kernelFuncBfloat16Conversion() {
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
  // CHECK: f2 = sycl::float2(sycl::ext::intel::math::bfloat162float(bf162[0]), sycl::ext::intel::math::bfloat162float(bf162[1]));
  f2 = __bfloat1622float2(bf162);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf16, bf16);
  bf162 = __bfloat162bfloat162(bf16);
  // CHECK: f = sycl::ext::intel::math::bfloat162float(bf16);
  f = __bfloat162float(bf16);
  // CHECK: i = sycl::ext::intel::math::bfloat162int_rd(bf16);
  i = __bfloat162int_rd(bf16);
  // CHECK: i = sycl::ext::intel::math::bfloat162int_rn(bf16);
  i = __bfloat162int_rn(bf16);
  // CHECK: i = sycl::ext::intel::math::bfloat162int_ru(bf16);
  i = __bfloat162int_ru(bf16);
  // CHECK: i = sycl::ext::intel::math::bfloat162int_rz(bf16);
  i = __bfloat162int_rz(bf16);
  // CHECK: ll = sycl::ext::intel::math::bfloat162ll_rd(bf16);
  ll = __bfloat162ll_rd(bf16);
  // CHECK: ll = sycl::ext::intel::math::bfloat162ll_rn(bf16);
  ll = __bfloat162ll_rn(bf16);
  // CHECK: ll = sycl::ext::intel::math::bfloat162ll_ru(bf16);
  ll = __bfloat162ll_ru(bf16);
  // CHECK: ll = sycl::ext::intel::math::bfloat162ll_rz(bf16);
  ll = __bfloat162ll_rz(bf16);
  // CHECK: s = sycl::ext::intel::math::bfloat162short_rd(bf16);
  s = __bfloat162short_rd(bf16);
  // CHECK: s = sycl::ext::intel::math::bfloat162short_rn(bf16);
  s = __bfloat162short_rn(bf16);
  // CHECK: s = sycl::ext::intel::math::bfloat162short_ru(bf16);
  s = __bfloat162short_ru(bf16);
  // CHECK: s = sycl::ext::intel::math::bfloat162short_rz(bf16);
  s = __bfloat162short_rz(bf16);
  // CHECK: u = sycl::ext::intel::math::bfloat162uint_rd(bf16);
  u = __bfloat162uint_rd(bf16);
  // CHECK: u = sycl::ext::intel::math::bfloat162uint_rn(bf16);
  u = __bfloat162uint_rn(bf16);
  // CHECK: u = sycl::ext::intel::math::bfloat162uint_ru(bf16);
  u = __bfloat162uint_ru(bf16);
  // CHECK: u = sycl::ext::intel::math::bfloat162uint_rz(bf16);
  u = __bfloat162uint_rz(bf16);
  // CHECK: ull = sycl::ext::intel::math::bfloat162ull_rd(bf16);
  ull = __bfloat162ull_rd(bf16);
  // CHECK: ull = sycl::ext::intel::math::bfloat162ull_rn(bf16);
  ull = __bfloat162ull_rn(bf16);
  // CHECK: ull = sycl::ext::intel::math::bfloat162ull_ru(bf16);
  ull = __bfloat162ull_ru(bf16);
  // CHECK: ull = sycl::ext::intel::math::bfloat162ull_rz(bf16);
  ull = __bfloat162ull_rz(bf16);
  // CHECK: us = sycl::ext::intel::math::bfloat162ushort_rd(bf16);
  us = __bfloat162ushort_rd(bf16);
  // CHECK: us = sycl::ext::intel::math::bfloat162ushort_rn(bf16);
  us = __bfloat162ushort_rn(bf16);
  // CHECK: us = sycl::ext::intel::math::bfloat162ushort_ru(bf16);
  us = __bfloat162ushort_ru(bf16);
  // CHECK: us = sycl::ext::intel::math::bfloat162ushort_rz(bf16);
  us = __bfloat162ushort_rz(bf16);
  // CHECK: s = sycl::ext::intel::math::bfloat16_as_short(bf16);
  s = __bfloat16_as_short(bf16);
  // CHECK: us = sycl::ext::intel::math::bfloat16_as_ushort(bf16);
  us = __bfloat16_as_ushort(bf16);
  // CHECK: bf16 = sycl::ext::intel::math::double2bfloat16(d);
  bf16 = __double2bfloat16(d);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f2[0], f2[1]);
  bf162 = __float22bfloat162_rn(f2);
  // CHECK: bf16 = sycl::ext::intel::math::float2bfloat16(f);
  bf16 = __float2bfloat16(f);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f, f);
  bf162 = __float2bfloat162_rn(f);
  // CHECK: bf16 = sycl::ext::intel::math::float2bfloat16_rd(f);
  bf16 = __float2bfloat16_rd(f);
  // CHECK: bf16 = sycl::ext::intel::math::float2bfloat16_rn(f);
  bf16 = __float2bfloat16_rn(f);
  // CHECK: bf16 = sycl::ext::intel::math::float2bfloat16_ru(f);
  bf16 = __float2bfloat16_ru(f);
  // CHECK: bf16 = sycl::ext::intel::math::float2bfloat16_rz(f);
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
  // CHECK: bf16 = sycl::ext::intel::math::int2bfloat16_rd(i);
  bf16 = __int2bfloat16_rd(i);
  // CHECK: bf16 = sycl::ext::intel::math::int2bfloat16_rn(i);
  bf16 = __int2bfloat16_rn(i);
  // CHECK: bf16 = sycl::ext::intel::math::int2bfloat16_ru(i);
  bf16 = __int2bfloat16_ru(i);
  // CHECK: bf16 = sycl::ext::intel::math::int2bfloat16_rz(i);
  bf16 = __int2bfloat16_rz(i);
  // CHECK: bf16 = sycl::ext::intel::math::ll2bfloat16_rd(ll);
  bf16 = __ll2bfloat16_rd(ll);
  // CHECK: bf16 = sycl::ext::intel::math::ll2bfloat16_rn(ll);
  bf16 = __ll2bfloat16_rn(ll);
  // CHECK: bf16 = sycl::ext::intel::math::ll2bfloat16_ru(ll);
  bf16 = __ll2bfloat16_ru(ll);
  // CHECK: bf16 = sycl::ext::intel::math::ll2bfloat16_rz(ll);
  bf16 = __ll2bfloat16_rz(ll);
  // CHECK: bf16 = sycl::ext::oneapi::bfloat16(bf162[0]);
  bf16 = __low2bfloat16(bf162);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf162[0], bf162[0]);
  bf162 = __low2bfloat162(bf162);
  // CHECK: bf162 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>(bf162_1[0], bf162_2[0]);
  bf162 = __lows2bfloat162(bf162_1, bf162_2);
  // CHECK: bf16 = sycl::ext::intel::math::short2bfloat16_rd(s);
  bf16 = __short2bfloat16_rd(s);
  // CHECK: bf16 = sycl::ext::intel::math::short2bfloat16_rn(s);
  bf16 = __short2bfloat16_rn(s);
  // CHECK: bf16 = sycl::ext::intel::math::short2bfloat16_ru(s);
  bf16 = __short2bfloat16_ru(s);
  // CHECK: bf16 = sycl::ext::intel::math::short2bfloat16_rz(s);
  bf16 = __short2bfloat16_rz(s);
  // CHECK: bf16 = sycl::ext::intel::math::short_as_bfloat16(s);
  bf16 = __short_as_bfloat16(s);
  // CHECK: bf16 = sycl::ext::intel::math::uint2bfloat16_rd(u);
  bf16 = __uint2bfloat16_rd(u);
  // CHECK: bf16 = sycl::ext::intel::math::uint2bfloat16_rn(u);
  bf16 = __uint2bfloat16_rn(u);
  // CHECK: bf16 = sycl::ext::intel::math::uint2bfloat16_ru(u);
  bf16 = __uint2bfloat16_ru(u);
  // CHECK: bf16 = sycl::ext::intel::math::uint2bfloat16_rz(u);
  bf16 = __uint2bfloat16_rz(u);
  // CHECK: bf16 = sycl::ext::intel::math::ull2bfloat16_rd(ull);
  bf16 = __ull2bfloat16_rd(ull);
  // CHECK: bf16 = sycl::ext::intel::math::ull2bfloat16_rn(ull);
  bf16 = __ull2bfloat16_rn(ull);
  // CHECK: bf16 = sycl::ext::intel::math::ull2bfloat16_ru(ull);
  bf16 = __ull2bfloat16_ru(ull);
  // CHECK: bf16 = sycl::ext::intel::math::ull2bfloat16_rz(ull);
  bf16 = __ull2bfloat16_rz(ull);
  // CHECK: bf16 = sycl::ext::intel::math::ushort2bfloat16_rd(us);
  bf16 = __ushort2bfloat16_rd(us);
  // CHECK: bf16 = sycl::ext::intel::math::ushort2bfloat16_rn(us);
  bf16 = __ushort2bfloat16_rn(us);
  // CHECK: bf16 = sycl::ext::intel::math::ushort2bfloat16_ru(us);
  bf16 = __ushort2bfloat16_ru(us);
  // CHECK: bf16 = sycl::ext::intel::math::ushort2bfloat16_rz(us);
  bf16 = __ushort2bfloat16_rz(us);
  // CHECK: bf16 = sycl::ext::intel::math::ushort_as_bfloat16(us);
  bf16 = __ushort_as_bfloat16(us);
}

int main() { return 0; }
