// RUN: dpct --format-range=none -out-root %T/enable-all-extensions %s --cuda-include-path="%cuda-path/include" --use-dpcpp-extensions=all
// RUN: FileCheck --input-file %T/enable-all-extensions/enable-all-extensions.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/enable-all-extensions/enable-all-extensions.dp.cpp -o %T/enable-all-extensions/enable-all-extensions.dp.o %}

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// c_cxx_standard_library

__device__ void d() {
  float f, f2;
  double d, d2;
  int i, i2;
  long l, l2;
  long long ll, ll2;

  // CHECK: f = sycl::fabs(f2);
  // CHECK-NEXT: d = sycl::fabs(d2);
  // CHECK-NEXT: i = sycl::abs(i2);
  // CHECK-NEXT: l = sycl::abs(l2);
  // CHECK-NEXT: ll = sycl::abs(ll2);
  // CHECK-NEXT: f = sycl::fabs(f2);
  // CHECK-NEXT: d = sycl::fabs(d2);
  // CHECK-NEXT: i = sycl::abs(i2);
  // CHECK-NEXT: l = sycl::abs(l2);
  // CHECK-NEXT: ll = sycl::abs(ll2);
  f = abs(f2);
  d = abs(d2);
  i = abs(i2);
  l = abs(l2);
  ll = abs(ll2);
  f = std::abs(f2);
  d = std::abs(d2);
  i = std::abs(i2);
  l = std::abs(l2);
  ll = std::abs(ll2);
}

void h() {
  float f, f2;
  double d, d2;
  int i, i2;
  long l, l2;
  long long ll, ll2;

  // CHECK: f = abs(f2);
  // CHECK-NEXT: d = abs(d2);
  // CHECK-NEXT: i = abs(i2);
  // CHECK-NEXT: l = abs(l2);
  // CHECK-NEXT: ll = abs(ll2);
  // CHECK-NEXT: f = std::abs(f2);
  // CHECK-NEXT: d = std::abs(d2);
  // CHECK-NEXT: i = std::abs(i2);
  // CHECK-NEXT: l = std::abs(l2);
  // CHECK-NEXT: ll = std::abs(ll2);
  f = abs(f2);
  d = abs(d2);
  i = abs(i2);
  l = abs(l2);
  ll = abs(ll2);
  f = std::abs(f2);
  d = std::abs(d2);
  i = std::abs(i2);
  l = std::abs(l2);
  ll = std::abs(ll2);
}

void foo1() {
  int n;
  // CHECK: dpct::dim3 abc;
  // CHECK-NEXT: abc.y = std::min(std::max(512 / abc.x, 1u), (unsigned int)n);
  // CHECK-NEXT: abc.z = std::min(std::max(512 / (abc.x * abc.y), 1u), (unsigned int)n);
  dim3 abc;
  abc.y = std::min(std::max(512 / abc.x, 1u), (unsigned int)n);
  abc.z = std::min(std::max(512 / (abc.x * abc.y), 1u), (unsigned int)n);
}

// intel_device_math
__global__ void kernelFuncHalfConversion() {
  float f;
  float2 f2;
  half h;
  half2 h2;
  int i;
  long long ll;
  short s;
  unsigned u;
  unsigned long long ull;
  unsigned short us;
  // CHECK: h2 = sycl::half2(sycl::ext::intel::math::float2half_rn(f2[0]), sycl::ext::intel::math::float2half_rn(f2[1]));
  h2 = __float22half2_rn(f2);
  // CHECK: h = sycl::ext::intel::math::float2half_rn(f);
  h = __float2half(f);
  // CHECK: h2 = sycl::half2(sycl::ext::intel::math::float2half_rn(f));
  h2 = __float2half2_rn(f);
  // CHECK: h = sycl::ext::intel::math::float2half_rd(f);
  h = __float2half_rd(f);
  // sycl::ext::intel::math::float2half_rn(f);
  __float2half_rn(f);
  // CHECK: h = sycl::ext::intel::math::float2half_ru(f);
  h = __float2half_ru(f);
  // CHECK: h = sycl::ext::intel::math::float2half_rz(f);
  h = __float2half_rz(f);
  // CHECK: h2 = sycl::half2(sycl::ext::intel::math::float2half_rn(f), sycl::ext::intel::math::float2half_rn(f));
  h2 = __floats2half2_rn(f, f);
  // CHECK: f2 = sycl::float2(sycl::ext::intel::math::half2float(h2[0]), sycl::ext::intel::math::half2float(h2[1]));
  f2 = __half22float2(h2);
  // CHECK: f = sycl::ext::intel::math::half2float(h);
  f = __half2float(h);
  // CHECK: h2 = sycl::half2(h);
  h2 = __half2half2(h);
  // CHECK: i = sycl::ext::intel::math::half2int_rd(h);
  i = __half2int_rd(h);
  // CHECK: i = sycl::ext::intel::math::half2int_rn(h);
  i = __half2int_rn(h);
  // CHECK: i = sycl::ext::intel::math::half2int_ru(h);
  i = __half2int_ru(h);
  // CHECK: i = sycl::ext::intel::math::half2int_rz(h);
  i = __half2int_rz(h);
  // CHECK: ll = sycl::ext::intel::math::half2ll_rd(h);
  ll = __half2ll_rd(h);
  // CHECK: ll = sycl::ext::intel::math::half2ll_rn(h);
  ll = __half2ll_rn(h);
  // CHECK: ll = sycl::ext::intel::math::half2ll_ru(h);
  ll = __half2ll_ru(h);
  // CHECK: ll = sycl::ext::intel::math::half2ll_rz(h);
  ll = __half2ll_rz(h);
  // CHECK: s = sycl::ext::intel::math::half2short_rd(h);
  s = __half2short_rd(h);
  // CHECK: s = sycl::ext::intel::math::half2short_rn(h);
  s = __half2short_rn(h);
  // CHECK: s = sycl::ext::intel::math::half2short_ru(h);
  s = __half2short_ru(h);
  // CHECK: s = sycl::ext::intel::math::half2short_rz(h);
  s = __half2short_rz(h);
  // CHECK: u = sycl::ext::intel::math::half2uint_rd(h);
  u = __half2uint_rd(h);
  // CHECK: u = sycl::ext::intel::math::half2uint_rn(h);
  u = __half2uint_rn(h);
  // CHECK: u = sycl::ext::intel::math::half2uint_ru(h);
  u = __half2uint_ru(h);
  // CHECK: u = sycl::ext::intel::math::half2uint_rz(h);
  u = __half2uint_rz(h);
  // CHECK: ull = sycl::ext::intel::math::half2ull_rd(h);
  ull = __half2ull_rd(h);
  // CHECK: ull = sycl::ext::intel::math::half2ull_rn(h);
  ull = __half2ull_rn(h);
  // CHECK: ull = sycl::ext::intel::math::half2ull_ru(h);
  ull = __half2ull_ru(h);
  // CHECK: ull = sycl::ext::intel::math::half2ull_rz(h);
  ull = __half2ull_rz(h);
  // CHECK: us = sycl::ext::intel::math::half2ushort_rd(h);
  us = __half2ushort_rd(h);
  // CHECK: us = sycl::ext::intel::math::half2ushort_rn(h);
  us = __half2ushort_rn(h);
  // CHECK: us = sycl::ext::intel::math::half2ushort_ru(h);
  us = __half2ushort_ru(h);
  // CHECK: us = sycl::ext::intel::math::half2ushort_rz(h);
  us = __half2ushort_rz(h);
  // CHECK: s = sycl::bit_cast<short, sycl::half>(h);
  s = __half_as_short(h);
  // CHECK: us = sycl::bit_cast<unsigned short, sycl::half>(h);
  us = __half_as_ushort(h);
  // CHECK: h2 = sycl::half2(h, h);
  h2 = __halves2half2(h, h);
  // CHECK: f = h2[1];
  f = __high2float(h2);
  // CHECK: h = h2[1];
  h = __high2half(h2);
  // CHECK: h2 = sycl::half2(h2[1]);
  h2 = __high2half2(h2);
  // CHECK: h2 = sycl::half2(h2[1], h2[1]);
  h2 = __highs2half2(h2, h2);
  // CHECK: h = sycl::ext::intel::math::int2half_rd(i);
  h = __int2half_rd(i);
  // CHECK: h = sycl::ext::intel::math::int2half_rn(i);
  h = __int2half_rn(i);
  // CHECK: h = sycl::ext::intel::math::int2half_ru(i);
  h = __int2half_ru(i);
  // CHECK: h = sycl::ext::intel::math::int2half_rz(i);
  h = __int2half_rz(i);
  // CHECK: h = sycl::ext::intel::math::ll2half_rd(ll);
  h = __ll2half_rd(ll);
  // CHECK: h = sycl::ext::intel::math::ll2half_rn(ll);
  h = __ll2half_rn(ll);
  // CHECK: h = sycl::ext::intel::math::ll2half_ru(ll);
  h = __ll2half_ru(ll);
  // CHECK: h = sycl::ext::intel::math::ll2half_rz(ll);
  h = __ll2half_rz(ll);
  // CHECK: f = h2[0];
  f = __low2float(h2);
  // CHECK: f = (*(&h2))[0];
  f = __low2float(*(&h2));
  // CHECK: h = h2[0];
  h = __low2half(h2);
  // CHECK: h2 = sycl::half2(h2[0]);
  h2 = __low2half2(h2);
  // CHECK: h2 = sycl::half2(h2[1], h2[0]);
  h2 = __lowhigh2highlow(h2);
  // CHECK: h2 = sycl::half2(h2[0], h2[0]);
  h2 = __lows2half2(h2, h2);
  // CHECK: h = sycl::ext::intel::math::short2half_rd(s);
  h = __short2half_rd(s);
  // CHECK: h = sycl::ext::intel::math::short2half_rn(s);
  h = __short2half_rn(s);
  // CHECK: h = sycl::ext::intel::math::short2half_ru(s);
  h = __short2half_ru(s);
  // CHECK: h = sycl::ext::intel::math::short2half_rz(s);
  h = __short2half_rz(s);
  // CHECK: h = sycl::bit_cast<sycl::half, short>(s);
  h = __short_as_half(s);
  // CHECK: h = sycl::ext::intel::math::uint2half_rd(u);
  h = __uint2half_rd(u);
  // CHECK: h = sycl::ext::intel::math::uint2half_rn(u);
  h = __uint2half_rn(u);
  // CHECK: h = sycl::ext::intel::math::uint2half_ru(u);
  h = __uint2half_ru(u);
  // CHECK: h = sycl::ext::intel::math::uint2half_rz(u);
  h = __uint2half_rz(u);
  // CHECK: h = sycl::ext::intel::math::ull2half_rd(ull);
  h = __ull2half_rd(ull);
  // CHECK: h = sycl::ext::intel::math::ull2half_rn(ull);
  h = __ull2half_rn(ull);
  // CHECK: h = sycl::ext::intel::math::ull2half_ru(ull);
  h = __ull2half_ru(ull);
  // CHECK: h = sycl::ext::intel::math::ull2half_rz(ull);
  h = __ull2half_rz(ull);
  // CHECK: h = sycl::ext::intel::math::ushort2half_rd(us);
  h = __ushort2half_rd(us);
  // CHECK: h = sycl::ext::intel::math::ushort2half_rn(us);
  h = __ushort2half_rn(us);
  // CHECK: h = sycl::ext::intel::math::ushort2half_ru(us);
  h = __ushort2half_ru(us);
  // CHECK: h = sycl::ext::intel::math::ushort2half_rz(us);
  h = __ushort2half_rz(us);
  // CHECK: h = sycl::bit_cast<sycl::half, unsigned short>(us);
  h = __ushort_as_half(us);
}
