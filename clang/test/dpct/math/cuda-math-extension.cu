// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/math/cuda-math-extension %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-extension/cuda-math-extension.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/ext/intel/math.hpp>
#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncDouble(double *deviceArrayDouble) {
  double &d0 = *deviceArrayDouble, &d1 = *(deviceArrayDouble + 1), &d2 = *(deviceArrayDouble + 2);
  int i;
  float f0;

  // Double Precision Mathematical Functions

  // CHECK: d2 = sycl::ext::intel::math::cyl_bessel_i0((double)f0);
  d2 = cyl_bessel_i0(f0);
  // CHECK: d2 = sycl::ext::intel::math::cyl_bessel_i0(d0);
  d2 = cyl_bessel_i0(d0);
  // CHECK: d2 = sycl::ext::intel::math::cyl_bessel_i1((double)f0);
  d2 = cyl_bessel_i1(f0);
  // CHECK: d2 = sycl::ext::intel::math::cyl_bessel_i1(d0);
  d2 = cyl_bessel_i1(d0);
  // CHECK: d2 = sycl::ext::intel::math::erfcinv(d0);
  d2 = erfcinv(d0);
  // CHECK: d2 = sycl::ext::intel::math::erfinv(d0);
  d2 = erfinv(d0);
  // CHECK: d2 = sycl::ext::intel::math::j0((double)f0);
  d2 = j0(f0);
  // CHECK: d2 = sycl::ext::intel::math::j0(d0);
  d2 = j0(d0);
  // CHECK: d2 = sycl::ext::intel::math::j1((double)f0);
  d2 = j1(f0);
  // CHECK: d2 = sycl::ext::intel::math::j1(d0);
  d2 = j1(d0);
  // CHECK: d2 = sycl::ext::intel::math::cdfnorm(d0);
  d2 = normcdf(d0);
  // CHECK: d2 = sycl::ext::intel::math::cdfnorminv(d0);
  d2 = normcdfinv(d0);
  // CHECK: d2 = sycl::ext::intel::math::norm(i, &d0);
  d2 = norm(i, &d0);
  // CHECK: d2 = sycl::ext::intel::math::norm(d1, &d0);
  d2 = norm(d1, &d0);
  // CHECK: d2 = sycl::ext::intel::math::rnorm(i, &d0);
  d2 = rnorm(i, &d0);
  // CHECK: d2 = sycl::ext::intel::math::rnorm(d1, &d0);
  d2 = rnorm(d1, &d0);
  // CHECK: d2 = sycl::ext::intel::math::y0((double)f0);
  d2 = y0(f0);
  // CHECK: d2 = sycl::ext::intel::math::y0(d0);
  d2 = y0(d0);
  // CHECK: d2 = sycl::ext::intel::math::y1((double)f0);
  d2 = y1(f0);
  // CHECK: d2 = sycl::ext::intel::math::y1(d0);
  d2 = y1(d0);

  // Double Precision Intrinsics

  // CHECK: d2 = sycl::ext::intel::math::dadd_rd(d0, d1);
  d2 = __dadd_rd(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dadd_rn(d0, d1);
  d2 = __dadd_rn(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dadd_ru(d0, d1);
  d2 = __dadd_ru(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dadd_rz(d0, d1);
  d2 = __dadd_rz(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dmul_rd(d0, d1);
  d2 = __dmul_rd(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dmul_rn(d0, d1);
  d2 = __dmul_rn(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dmul_ru(d0, d1);
  d2 = __dmul_ru(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dmul_rz(d0, d1);
  d2 = __dmul_rz(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dsub_rd(d0, d1);
  d2 = __dsub_rd(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dsub_rn(d0, d1);
  d2 = __dsub_rn(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dsub_ru(d0, d1);
  d2 = __dsub_ru(d0, d1);
  // CHECK: d2 = sycl::ext::intel::math::dsub_rz(d0, d1);
  d2 = __dsub_rz(d0, d1);
}

__global__ void kernelFuncFloat(float *deviceArrayFloat) {
  float &f0 = *deviceArrayFloat, &f1 = *(deviceArrayFloat + 1), &f2 = *(deviceArrayFloat + 2);
  int i;
  double d0;

  // Single Precision Mathematical Functions

  // CHECK: f2 = sycl::ext::intel::math::cyl_bessel_i0(f0);
  f2 = cyl_bessel_i0f(f0);
  // CHECK: f2 = sycl::ext::intel::math::cyl_bessel_i0((float)d0);
  f2 = cyl_bessel_i0f(d0);
  // CHECK: f2 = sycl::ext::intel::math::cyl_bessel_i1(f0);
  f2 = cyl_bessel_i1f(f0);
  // CHECK: f2 = sycl::ext::intel::math::cyl_bessel_i1((float)d0);
  f2 = cyl_bessel_i1f(d0);
  // CHECK: f2 = sycl::ext::intel::math::erfcinv(f0);
  f2 = erfcinvf(f0);
  // CHECK: f2 = sycl::ext::intel::math::erfcinv((float)i);
  f2 = erfcinvf(i);
  // CHECK: f2 = sycl::ext::intel::math::erfinv(f0);
  f2 = erfinvf(f0);
  // CHECK: f2 = sycl::ext::intel::math::erfinv((float)i);
  f2 = erfinvf(i);
  // CHECK: f2 = sycl::ext::intel::math::j0(f0);
  f2 = j0f(f0);
  // CHECK: f2 = sycl::ext::intel::math::j0((float)d0);
  f2 = j0f(d0);
  // CHECK: f2 = sycl::ext::intel::math::j1(f0);
  f2 = j1f(f0);
  // CHECK: f2 = sycl::ext::intel::math::j1((float)d0);
  f2 = j1f(d0);
  // CHECK: f2 = sycl::ext::intel::math::cdfnorm(f0);
  f2 = normcdff(f0);
  // CHECK: f2 = sycl::ext::intel::math::cdfnorm((float)i);
  f2 = normcdff(i);
  // CHECK: f2 = sycl::ext::intel::math::cdfnorminv(f0);
  f2 = normcdfinvf(f0);
  // CHECK: f2 = sycl::ext::intel::math::cdfnorminv((float)i);
  f2 = normcdfinvf(i);
  // CHECK: f2 = sycl::ext::intel::math::norm(i, &f0);
  f2 = normf(i, &f0);
  // CHECK: f2 = sycl::ext::intel::math::norm(f1, &f0);
  f2 = normf(f1, &f0);
  // CHECK: f2 = sycl::ext::intel::math::rnorm(i, &f0);
  f2 = rnormf(i, &f0);
  // CHECK: f2 = sycl::ext::intel::math::rnorm(f1, &f0);
  f2 = rnormf(f1, &f0);
  // CHECK: f2 = sycl::ext::intel::math::y0(f0);
  f2 = y0f(f0);
  // CHECK: f2 = sycl::ext::intel::math::y0((float)d0);
  f2 = y0f(d0);
  // CHECK: f2 = sycl::ext::intel::math::y1(f0);
  f2 = y1f(f0);
  // CHECK: f2 = sycl::ext::intel::math::y1((float)d0);
  f2 = y1f(d0);

  // Single Precision Intrinsics

  // CHECK: f2 = sycl::ext::intel::math::fadd_rd(f0, f1);
  f2 = __fadd_rd(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fadd_rn(f0, f1);
  f2 = __fadd_rn(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fadd_ru(f0, f1);
  f2 = __fadd_ru(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fadd_rz(f0, f1);
  f2 = __fadd_rz(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fmul_rd(f0, f1);
  f2 = __fmul_rd(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fmul_rn(f0, f1);
  f2 = __fmul_rn(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fmul_ru(f0, f1);
  f2 = __fmul_ru(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fmul_rz(f0, f1);
  f2 = __fmul_rz(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fsub_rd(f0, f1);
  f2 = __fsub_rd(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fsub_rn(f0, f1);
  f2 = __fsub_rn(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fsub_ru(f0, f1);
  f2 = __fsub_ru(f0, f1);
  // CHECK: f2 = sycl::ext::intel::math::fsub_rz(f0, f1);
  f2 = __fsub_rz(f0, f1);
}

__global__ void kernelFuncHalf() {
  __half h, h_1, h_2;
  bool b;

  // Half Arithmetic Functions

  // CHECK: h_2 = sycl::ext::intel::math::hadd_sat(h, h_1);
  h_2 = __hadd_sat(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hfma_sat(h, h_1, h_2);
  h_2 = __hfma_sat(h, h_1, h_2);
  // CHECK: h_2 = sycl::ext::intel::math::hmul(h, h_1);
  h_2 = __hmul(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hmul_sat(h, h_1);
  h_2 = __hmul_sat(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hneg(h_1);
  h_2 = __hneg(h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hsub(h, h_1);
  h_2 = __hsub(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hsub_sat(h, h_1);
  h_2 = __hsub_sat(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hadd(h, h_1);
  h_2 = __hadd(h, h_1);

  // Half Comparison Functions

  // CHECK: b = sycl::ext::intel::math::heq(h, h_1);
  b = __heq(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hequ(h, h_1);
  b = __hequ(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hge(h, h_1);
  b = __hge(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hgeu(h, h_1);
  b = __hgeu(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hgt(h, h_1);
  b = __hgt(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hgtu(h, h_1);
  b = __hgtu(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hisinf(h);
  b = __hisinf(h);
  // CHECK: b = sycl::ext::intel::math::hisnan(h);
  b = __hisnan(h);
  // CHECK: b = sycl::ext::intel::math::hle(h, h_1);
  b = __hle(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hleu(h, h_1);
  b = __hleu(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hlt(h, h_1);
  b = __hlt(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hltu(h, h_1);
  b = __hltu(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hne(h, h_1);
  b = __hne(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hneu(h, h_1);
  b = __hneu(h, h_1);

  // Half Math Functions

  // CHECK: h_2 = sycl::ext::intel::math::ceil(h);
  h_2 = hceil(h);
  // CHECK: h_2 = sycl::ext::intel::math::cos(h);
  h_2 = hcos(h);
  // CHECK: h_2 = sycl::ext::intel::math::exp(h);
  h_2 = hexp(h);
  // CHECK: h_2 = sycl::ext::intel::math::exp10(h);
  h_2 = hexp10(h);
  // CHECK: h_2 = sycl::ext::intel::math::exp2(h);
  h_2 = hexp2(h);
  // CHECK: h_2 = sycl::ext::intel::math::floor(h);
  h_2 = hfloor(h);
  // CHECK: h_2 = sycl::ext::intel::math::log(h);
  h_2 = hlog(h);
  // CHECK: h_2 = sycl::ext::intel::math::log10(h);
  h_2 = hlog10(h);
  // CHECK: h_2 = sycl::ext::intel::math::log2(h);
  h_2 = hlog2(h);
  // CHECK: h_2 = sycl::ext::intel::math::inv(h);
  h_2 = hrcp(h);
  // CHECK: h_2 = sycl::ext::intel::math::rint(h);
  h_2 = hrint(h);
  // CHECK: h_2 = sycl::ext::intel::math::rsqrt(h);
  h_2 = hrsqrt(h);
  // CHECK: h_2 = sycl::ext::intel::math::sin(h);
  h_2 = hsin(h);
  // CHECK: h_2 = sycl::ext::intel::math::sqrt(h);
  h_2 = hsqrt(h);
  // CHECK: h_2 = sycl::ext::intel::math::trunc(h);
  h_2 = htrunc(h);
}

__global__ void kernelFuncHalf2() {
  __half2 h2, h2_1, h2_2;
  bool b;

  // Half2 Arithmetic Functions

  // CHECK: h2_2 = sycl::ext::intel::math::h2div(h2, h2_1);
  h2_2 = __h2div(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hadd2(h2, h2_1);
  h2_2 = __hadd2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hadd2_sat(h2, h2_1);
  h2_2 = __hadd2_sat(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hfma2_sat(h2, h2_1, h2_2);
  h2_2 = __hfma2_sat(h2, h2_1, h2_2);
  // CHECK: h2_2 = sycl::ext::intel::math::hmul2(h2, h2_1);
  h2_2 = __hmul2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hmul2_sat(h2, h2_1);
  h2_2 = __hmul2_sat(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hneg2(h2_1);
  h2_2 = __hneg2(h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hsub2(h2, h2_1);
  h2_2 = __hsub2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hsub2_sat(h2, h2_1);
  h2_2 = __hsub2_sat(h2, h2_1);

  // Half2 Comparison Functions

  // CHECK: b = sycl::ext::intel::math::hbeq2(h2, h2_1);
  b = __hbeq2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbequ2(h2, h2_1);
  b = __hbequ2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbge2(h2, h2_1);
  b = __hbge2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbgeu2(h2, h2_1);
  b = __hbgeu2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbgt2(h2, h2_1);
  b = __hbgt2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbgtu2(h2, h2_1);
  b = __hbgtu2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hble2(h2, h2_1);
  b = __hble2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbleu2(h2, h2_1);
  b = __hbleu2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hblt2(h2, h2_1);
  b = __hblt2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbltu2(h2, h2_1);
  b = __hbltu2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbne2(h2, h2_1);
  b = __hbne2(h2, h2_1);
  // CHECK: b = sycl::ext::intel::math::hbneu2(h2, h2_1);
  b = __hbneu2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::heq2(h2, h2_1);
  h2_2 = __heq2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hequ2(h2, h2_1);
  h2_2 = __hequ2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hge2(h2, h2_1);
  h2_2 = __hge2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hgeu2(h2, h2_1);
  h2_2 = __hgeu2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hgt2(h2, h2_1);
  h2_2 = __hgt2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hgtu2(h2, h2_1);
  h2_2 = __hgtu2(h2, h2_1);

  // CHECK: h2_2 = sycl::ext::intel::math::hisnan2(h2);
  h2_2 = __hisnan2(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::hle2(h2, h2_1);
  h2_2 = __hle2(h2, h2_1);
  // CHECK: sycl::ext::intel::math::hleu2(h2, h2);
  __hleu2(h2, h2);
  // CHECK: h2_2 = sycl::ext::intel::math::hlt2(h2, h2_1);
  h2_2 = __hlt2(h2, h2_1);
  // CHECK: sycl::ext::intel::math::hltu2(h2, h2);
  __hltu2(h2, h2);
  // CHECK: h2_2 = sycl::ext::intel::math::hne2(h2, h2_1);
  h2_2 = __hne2(h2, h2_1);
  // CHECK: sycl::ext::intel::math::hneu2(h2, h2);
  __hneu2(h2, h2);

  // Half2 Math Functions

  // CHECK: h2_2 = sycl::ext::intel::math::ceil(h2);
  h2_2 = h2ceil(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::cos(h2);
  h2_2 = h2cos(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::exp(h2);
  h2_2 = h2exp(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::exp10(h2);
  h2_2 = h2exp10(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::exp2(h2);
  h2_2 = h2exp2(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::floor(h2);
  h2_2 = h2floor(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::log(h2);
  h2_2 = h2log(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::log10(h2);
  h2_2 = h2log10(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::log2(h2);
  h2_2 = h2log2(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::inv(h2);
  h2_2 = h2rcp(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::rint(h2);
  h2_2 = h2rint(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::rsqrt(h2);
  h2_2 = h2rsqrt(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::sin(h2);
  h2_2 = h2sin(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::sqrt(h2);
  h2_2 = h2sqrt(h2);
  // CHECK: h2_2 = sycl::ext::intel::math::trunc(h2);
  h2_2 = h2trunc(h2);
}

__global__ void kernelFuncInt() {
  int i;
  unsigned u;
  long l;
  long long ll;
  unsigned long long ull;
  // CHECK: ll = dpct::max(ll, ll);
  // CHECK-NEXT: ll = dpct::min(ll, ll);
  // CHECK-NEXT: ull = dpct::max(ull, ull);
  // CHECK-NEXT: ull = dpct::min(ull, ull);
  // CHECK-NEXT: u = dpct::max(u, u);
  // CHECK-NEXT: u = dpct::min(u, u);
  ll = llmax(ll, ll);
  ll = llmin(ll, ll);
  ull = ullmax(ull, ull);
  ull = ullmin(ull, ull);
  u = umax(u, u);
  u = umin(u, u);
}

__global__ void kernelFuncSIMD() {
  unsigned int u, u_1, u_2;

  // CHECK: u_2 = sycl::ext::intel::math::vabs2(u);
  u_2 = __vabs2(u);
  // CHECK: u_2 = sycl::ext::intel::math::vabs4(u);
  u_2 = __vabs4(u);
  // CHECK: u_2 = sycl::ext::intel::math::vabsdiffs2(u, u_1);
  u_2 = __vabsdiffs2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vabsdiffs4(u, u_1);
  u_2 = __vabsdiffs4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vabsdiffu2(u, u_1);
  u_2 = __vabsdiffu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vabsdiffu4(u, u_1);
  u_2 = __vabsdiffu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vabsss2(u);
  u_2 = __vabsss2(u);
  // CHECK: u_2 = sycl::ext::intel::math::vabsss4(u);
  u_2 = __vabsss4(u);
  // CHECK: u_2 = sycl::ext::intel::math::vadd2(u, u_1);
  u_2 = __vadd2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vadd4(u, u_1);
  u_2 = __vadd4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vaddss2(u, u_1);
  u_2 = __vaddss2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vaddss4(u, u_1);
  u_2 = __vaddss4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vaddus2(u, u_1);
  u_2 = __vaddus2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vaddus4(u, u_1);
  u_2 = __vaddus4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vavgs2(u, u_1);
  u_2 = __vavgs2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vavgs4(u, u_1);
  u_2 = __vavgs4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vavgu2(u, u_1);
  u_2 = __vavgu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vavgu4(u, u_1);
  u_2 = __vavgu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpeq2(u, u_1);
  u_2 = __vcmpeq2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpeq4(u, u_1);
  u_2 = __vcmpeq4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpges2(u, u_1);
  u_2 = __vcmpges2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpges4(u, u_1);
  u_2 = __vcmpges4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpgeu2(u, u_1);
  u_2 = __vcmpgeu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpgeu4(u, u_1);
  u_2 = __vcmpgeu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpgts2(u, u_1);
  u_2 = __vcmpgts2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpgts4(u, u_1);
  u_2 = __vcmpgts4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpgtu2(u, u_1);
  u_2 = __vcmpgtu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpgtu4(u, u_1);
  u_2 = __vcmpgtu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmples2(u, u_1);
  u_2 = __vcmples2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmples4(u, u_1);
  u_2 = __vcmples4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpleu2(u, u_1);
  u_2 = __vcmpleu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpleu4(u, u_1);
  u_2 = __vcmpleu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmplts2(u, u_1);
  u_2 = __vcmplts2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmplts4(u, u_1);
  u_2 = __vcmplts4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpltu2(u, u_1);
  u_2 = __vcmpltu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpltu4(u, u_1);
  u_2 = __vcmpltu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpne2(u, u_1);
  u_2 = __vcmpne2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vcmpne4(u, u_1);
  u_2 = __vcmpne4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vhaddu2(u, u_1);
  u_2 = __vhaddu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vhaddu4(u, u_1);
  u_2 = __vhaddu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vmaxs2(u, u_1);
  u_2 = __vmaxs2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vmaxs4(u, u_1);
  u_2 = __vmaxs4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vmaxu2(u, u_1);
  u_2 = __vmaxu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vmaxu4(u, u_1);
  u_2 = __vmaxu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vmins2(u, u_1);
  u_2 = __vmins2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vmins4(u, u_1);
  u_2 = __vmins4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vminu2(u, u_1);
  u_2 = __vminu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vminu4(u, u_1);
  u_2 = __vminu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vneg2(u);
  u_2 = __vneg2(u);
  // CHECK: u_2 = sycl::ext::intel::math::vneg4(u);
  u_2 = __vneg4(u);
  // CHECK: u_2 = sycl::ext::intel::math::vnegss2(u);
  u_2 = __vnegss2(u);
  // CHECK: u_2 = sycl::ext::intel::math::vnegss4(u);
  u_2 = __vnegss4(u);
  // CHECK: u_2 = sycl::ext::intel::math::vsads2(u, u_1);
  u_2 = __vsads2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsads4(u, u_1);
  u_2 = __vsads4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsadu2(u, u_1);
  u_2 = __vsadu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsadu4(u, u_1);
  u_2 = __vsadu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vseteq2(u, u_1);
  u_2 = __vseteq2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vseteq4(u, u_1);
  u_2 = __vseteq4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetges2(u, u_1);
  u_2 = __vsetges2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetges4(u, u_1);
  u_2 = __vsetges4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetgeu2(u, u_1);
  u_2 = __vsetgeu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetgeu4(u, u_1);
  u_2 = __vsetgeu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetgts2(u, u_1);
  u_2 = __vsetgts2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetgts4(u, u_1);
  u_2 = __vsetgts4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetgtu2(u, u_1);
  u_2 = __vsetgtu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetgtu4(u, u_1);
  u_2 = __vsetgtu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetles2(u, u_1);
  u_2 = __vsetles2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetles4(u, u_1);
  u_2 = __vsetles4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetleu2(u, u_1);
  u_2 = __vsetleu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetleu4(u, u_1);
  u_2 = __vsetleu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetlts2(u, u_1);
  u_2 = __vsetlts2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetlts4(u, u_1);
  u_2 = __vsetlts4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetltu2(u, u_1);
  u_2 = __vsetltu2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetltu4(u, u_1);
  u_2 = __vsetltu4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetne2(u, u_1);
  u_2 = __vsetne2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsetne4(u, u_1);
  u_2 = __vsetne4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsub2(u, u_1);
  u_2 = __vsub2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsub4(u, u_1);
  u_2 = __vsub4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsubss2(u, u_1);
  u_2 = __vsubss2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsubss4(u, u_1);
  u_2 = __vsubss4(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsubus2(u, u_1);
  u_2 = __vsubus2(u, u_1);
  // CHECK: u_2 = sycl::ext::intel::math::vsubus4(u, u_1);
  u_2 = __vsubus4(u, u_1);
}

int main() { return 0; }
