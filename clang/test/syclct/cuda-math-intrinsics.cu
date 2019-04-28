// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cuda-math-intrinsics.sycl.cpp --match-full-lines %s

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <stdio.h>

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncHalf(double *deviceArrayDouble) {
  __half h, h_1, h_2;
  __half2 h2, h2_1, h2_2;
  bool b;

  // Half Arithmetic Functions

  // TODO:1CHECK: h2_2 = h2 / h2_1;
  //h2_2 = __h2div(h2, h2_1);
  // TODO:1CHECK: h_2 = h / h_1;
  //h_2 = __hdiv(h, h_1);
  // CHECK: h_2 = cl::sycl::fma(h, h_1, h_2);
  h_2 = __hfma(h, h_1, h_2);
  // CHECK: h_2 = h * h_1;
  h_2 = __hmul(h, h_1);
  // CHECK: h_2 = -h;
  h_2 = __hneg(h);
  // CHECK: h_2 = h - h_1;
  h_2 = __hsub(h, h_1);

  // Half2 Arithmetic Functions

  // CHECK: h2_2 = cl::sycl::fma(h2, h2_1, h2_2);
  h2_2 = __hfma2(h2, h2_1, h2_2);
  // CHECK: h2_2 = h2 * h2_1;
  h2_2 = __hmul2(h2, h2_1);
  // CHECK: h2_2 = -h2;
  h2_2 = __hneg2(h2);
  // CHECK: h2_2 = h2 - h2_1;
  h2_2 = __hsub2(h2, h2_1);

  // Half Comparison Functions

  // CHECK: b = h == h_1;
  b = __heq(h, h_1);
  // CHECK: b = h >= h_1;
  b = __hge(h, h_1);
  // CHECK: b = h > h_1;
  b = __hgt(h, h_1);
  // CHECK: b = cl::sycl::isinf(h);
  b = __hisinf(h);
  // CHECK: b = cl::sycl::isnan(h);
  b = __hisnan(h);
  // CHECK: b = h <= h_1;
  b = __hle(h, h_1);
  // CHECK: b = h < h_1;
  b = __hlt(h, h_1);
  // CHECK: b = h != h_1;
  b = __hne(h, h_1);

  // Half2 Comparison Functions

  // CHECK: h2_2 = h2 == h2_1;
  h2_2 = __heq2(h2, h2_1);
  // CHECK: h2_2 = h2 >= h2_1;
  h2_2 = __hge2(h2, h2_1);
  // CHECK: h2_2 = h2 > h2_1;
  h2_2 = __hgt2(h2, h2_1);
  // CHECK: h2_2 = cl::sycl::isnan(h2);
  h2_2 = __hisnan2(h2);
  // CHECK: h2_2 = h2 <= h2_1;
  h2_2 = __hle2(h2, h2_1);
  // CHECK: h2_2 = h2 < h2_1;
  h2_2 = __hlt2(h2, h2_1);
  // CHECK: h2_2 = h2 != h2_1;
  h2_2 = __hne2(h2, h2_1);

  // Half Math Functions

  // CHECK: h_2 = cl::sycl::ceil(h);
  h_2 = hceil(h);
  // CHECK: h_2 = cl::sycl::cos(h);
  h_2 = hcos(h);
  // CHECK: h_2 = cl::sycl::exp(h);
  h_2 = hexp(h);
  // CHECK: h_2 = cl::sycl::exp10(h);
  h_2 = hexp10(h);
  // CHECK: h_2 = cl::sycl::exp2(h);
  h_2 = hexp2(h);
  // CHECK: h_2 = cl::sycl::floor(h);
  h_2 = hfloor(h);
  // CHECK: h_2 = cl::sycl::log(h);
  h_2 = hlog(h);
  // CHECK: h_2 = cl::sycl::log10(h);
  h_2 = hlog10(h);
  // CHECK: h_2 = cl::sycl::log2(h);
  h_2 = hlog2(h);
  // CHECK: h_2 = cl::sycl::recip(h);
  h_2 = hrcp(h);
  // CHECK: h_2 = cl::sycl::rint(h);
  h_2 = hrint(h);
  // CHECK: h_2 = cl::sycl::rsqrt(h);
  h_2 = hrsqrt(h);
  // CHECK: h_2 = cl::sycl::sin(h);
  h_2 = hsin(h);
  // CHECK: h_2 = cl::sycl::sqrt(h);
  h_2 = hsqrt(h);
  // CHECK: h_2 = cl::sycl::trunc(h);
  h_2 = htrunc(h);

  // Half2 Math Functions

  // CHECK: h2_2 = cl::sycl::ceil(h2);
  h2_2 = h2ceil(h2);
  // CHECK: h2_2 = cl::sycl::cos(h2);
  h2_2 = h2cos(h2);
  // CHECK: h2_2 = cl::sycl::exp(h2);
  h2_2 = h2exp(h2);
  // CHECK: h2_2 = cl::sycl::exp10(h2);
  h2_2 = h2exp10(h2);
  // CHECK: h2_2 = cl::sycl::exp2(h2);
  h2_2 = h2exp2(h2);
  // CHECK: h2_2 = cl::sycl::floor(h2);
  h2_2 = h2floor(h2);
  // CHECK: h2_2 = cl::sycl::log(h2);
  h2_2 = h2log(h2);
  // CHECK: h2_2 = cl::sycl::log10(h2);
  h2_2 = h2log10(h2);
  // CHECK: h2_2 = cl::sycl::log2(h2);
  h2_2 = h2log2(h2);
  // CHECK: h2_2 = cl::sycl::recip(h2);
  h2_2 = h2rcp(h2);
  // CHECK: h2_2 = cl::sycl::rint(h2);
  h2_2 = h2rint(h2);
  // CHECK: h2_2 = cl::sycl::rsqrt(h2);
  h2_2 = h2rsqrt(h2);
  // CHECK: h2_2 = cl::sycl::sin(h2);
  h2_2 = h2sin(h2);
  // CHECK: h2_2 = cl::sycl::sqrt(h2);
  h2_2 = h2sqrt(h2);
  // CHECK: h2_2 = cl::sycl::trunc(h2);
  h2_2 = h2trunc(h2);
}

__global__ void kernelFuncDouble(double *deviceArrayDouble) {
  double &d0 = *deviceArrayDouble, &d1 = *(deviceArrayDouble + 1), &d2 = *(deviceArrayDouble + 2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 + d1;
  d2 = __dadd_rd(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 + d1;
  d2 = __dadd_rn(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 + d1;
  d2 = __dadd_ru(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 + d1;
  d2 = __dadd_rz(d0, d1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 / d1;
  d2 = __ddiv_rd(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 / d1;
  d2 = __ddiv_rn(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 / d1;
  d2 = __ddiv_ru(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 / d1;
  d2 = __ddiv_rz(d0, d1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 * d1;
  d2 = __dmul_rd(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 * d1;
  d2 = __dmul_rn(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 * d1;
  d2 = __dmul_ru(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 * d1;
  d2 = __dmul_rz(d0, d1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = cl::sycl::recip(d0);
  d1 = __frcp_rd(d0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = cl::sycl::recip(d0);
  d1 = __frcp_rn(d0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = cl::sycl::recip(d0);
  d1 = __drcp_ru(d0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = cl::sycl::recip(d0);
  d1 = __drcp_rz(d0);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d0 = cl::sycl::sqrt(d0);
  d0 = __dsqrt_rd(d0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = cl::sycl::sqrt(d1);
  d1 = __dsqrt_rn(d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d0 = cl::sycl::sqrt(d0);
  d0 = __dsqrt_ru(d0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = cl::sycl::sqrt(d1);
  d1 = __dsqrt_rz(d1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 - d1;
  d2 = __dsub_rd(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 - d1;
  d2 = __dsub_rn(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 - d1;
  d2 = __dsub_ru(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 - d1;
  d2 = __dsub_rz(d0, d1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = cl::sycl::fma(d0, d1, d2);
  d2 = __fma_rd(d0, d1, d2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = cl::sycl::fma(d0, d1, d2);
  d2 = __fma_rn(d0, d1, d2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = cl::sycl::fma(d0, d1, d2);
  d2 = __fma_ru(d0, d1, d2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = cl::sycl::fma(d0, d1, d2);
  d2 = __fma_rz(d0, d1, d2);

  // CHECK: d0 = cl::sycl::fmin(d0, d1);
  d0 = fmin(d0, d1);
  // CHECK: d2 = cl::sycl::fmax(d0, d1);
  d2 = fmax(d0, d1);

  // CHECK: d1 = cl::sycl::floor(d1);
  d1 = floor(d1);
  // CHECK: d2 = cl::sycl::ceil(d2);
  d2 = ceil(d2);

  // CHECK: d2 = cl::sycl::fma(d0, d1, d2);
  d2 = fma(d0, d1, d2);
  // CHECK: d2 = cl::sycl::nan(0);
  d2 = nan("NaN");
}

__global__ void kernelFuncFloat(float *deviceArrayFloat) {
  float &f0 = *deviceArrayFloat, &f1 = *(deviceArrayFloat + 1), &f2 = *(deviceArrayFloat + 2);
  int i;

  // Single Precision Mathematical Functions
  // CHECK: f2 = cl::sycl::log(f0);
  f2 = log(f0);
  // CHECK: f2 = cl::sycl::log(f0);
  f2 = logf(f0);
  // CHECK: f2 = cl::sycl::acos(f0);
  f2 = acosf(f0);
  // CHECK: f2 = cl::sycl::acosh(f0);
  f2 = acoshf(f0);
  // CHECK: f2 = cl::sycl::asin(f0);
  f2 = asinf(f0);
  // CHECK: f2 = cl::sycl::asinh(f0);
  f2 = asinhf(f0);
  // CHECK: f2 = cl::sycl::atan2(f0, f1);
  f2 = atan2f(f0, f1);
  // CHECK: f2 = cl::sycl::atan(f0);
  f2 = atanf(f0);
  // CHECK: f2 = cl::sycl::atanh(f0);
  f2 = atanhf(f0);
  // CHECK: f2 = cl::sycl::cbrt(f0);
  f2 = cbrtf(f0);
  // CHECK: f2 = cl::sycl::ceil(f0);
  f2 = ceilf(f0);
  // CHECK: f2 = cl::sycl::copysign(f0, f1);
  f2 = copysignf(f0, f1);
  // CHECK: f2 = cl::sycl::cos(f0);
  f2 = cosf(f0);
  // CHECK: f2 = cl::sycl::cosh(f0);
  f2 = coshf(f0);
  // CHECK: f2 = cl::sycl::cospi(f0);
  f2 = cospif(f0);
  // CHECK: f2 = cl::sycl::erfc(f0);
  f2 = erfcf(f0);
  // CHECK: f2 = cl::sycl::erf(f0);
  f2 = erff(f0);
  // CHECK: f2 = cl::sycl::exp10(f0);
  f2 = exp10f(f0);
  // CHECK: f2 = cl::sycl::exp2(f0);
  f2 = exp2f(f0);
  // CHECK: f2 = cl::sycl::exp(f0);
  f2 = expf(f0);
  // CHECK: f2 = cl::sycl::expm1(f0);
  f2 = expm1f(f0);
  // CHECK: f2 = cl::sycl::fabs(f0);
  f2 = fabsf(f0);
  // CHECK: f2 = cl::sycl::fdim(f0, f1);
  f2 = fdimf(f0, f1);
  // CHECK: f2 = cl::sycl::native::divide(f0, f1);
  f2 = fdividef(f0, f1);
  // CHECK: f2 = cl::sycl::floor(f0);
  f2 = floorf(f0);
  // CHECK: f2 = cl::sycl::fma(f0, f1, f2);
  f2 = fmaf(f0, f1, f2);
  // CHECK: f2 = cl::sycl::fmax(f0, f1);
  f2 = fmaxf(f0, f1);
  // CHECK: f2 = cl::sycl::fmin(f0, f1);
  f2 = fminf(f0, f1);
  // CHECK: f2 = cl::sycl::fmod(f0, f1);
  f2 = fmodf(f0, f1);
  // CHECK: f2 = cl::sycl::frexp(f0, &i);
  f2 = frexpf(f0, &i);
  // CHECK: f2 = cl::sycl::hypot(f0, f1);
  f2 = hypotf(f0, f1);
  // CHECK: f2 = cl::sycl::ilogb(f0);
  f2 = ilogbf(f0);
  // CHECK: f2 = cl::sycl::isfinite(f0);
  f2 = isfinite(f0);
  // CHECK: f2 = cl::sycl::isinf(f0);
  f2 = isinf(f0);
  // CHECK: f2 = cl::sycl::isnan(f0);
  f2 = isnan(f0);
  // CHECK: f2 = cl::sycl::ldexp(f0, i);
  f2 = ldexpf(f0, i);
  // CHECK: f2 = cl::sycl::lgamma(f0);
  f2 = lgammaf(f0);
  // CHECK: f2 = cl::sycl::rint(f0);
  f2 = llrintf(f0);
  // CHECK: f2 = cl::sycl::round(f0);
  f2 = llroundf(f0);
  // CHECK: f2 = cl::sycl::log10(f0);
  f2 = log10f(f0);
  // CHECK: f2 = cl::sycl::log1p(f0);
  f2 = log1pf(f0);
  // CHECK: f2 = cl::sycl::log2(f0);
  f2 = log2f(f0);
  // CHECK: f2 = cl::sycl::logb(f0);
  f2 = logbf(f0);
  // CHECK: f2 = cl::sycl::rint(f0);
  f2 = lrintf(f0);
  // CHECK: f2 = cl::sycl::round(f0);
  f2 = lroundf(f0);
  // CHECK: f2 = cl::sycl::modf(f0, &f1);
  f2 = modff(f0, &f1);
  // CHECK: f2 = cl::sycl::nan(0);
  f2 = nan("");
  // CHECK: f2 = cl::sycl::pow(f0, f1);
  f2 = powf(f0, f1);
  // CHECK: f2 = cl::sycl::remainder(f0, f1);
  f2 = remainderf(f0, f1);
  // CHECK: f2 = cl::sycl::remquo(f0, f1, &i);
  f2 = remquof(f0, f1, &i);
  // CHECK: f2 = cl::sycl::rint(f0);
  f2 = rintf(f0);
  // CHECK: f2 = cl::sycl::round(f0);
  f2 = roundf(f0);
  // CHECK: f2 = cl::sycl::rsqrt(f0);
  f2 = rsqrtf(f0);
  // CHECK: f2 = cl::sycl::signbit(f0);
  f2 = signbit(f0);
  // CHECK: *(&f1) = cl::sycl::sincos(f0, &f2);
  sincosf(f0, &f1, &f2);
  // CHECK: f2 = cl::sycl::sin(f0);
  f2 = sinf(f0);
  // CHECK: f2 = cl::sycl::sinh(f0);
  f2 = sinhf(f0);
  // CHECK: f2 = cl::sycl::sinpi(f0);
  f2 = sinpif(f0);
  // CHECK: f2 = cl::sycl::sqrt(f0);
  f2 = sqrtf(f0);
  // CHECK: f2 = cl::sycl::tan(f0);
  f2 = tanf(f0);
  // CHECK: f2 = cl::sycl::tanh(f0);
  f2 = tanhf(f0);
  // CHECK: f2 = cl::sycl::tgamma(f0);
  f2 = tgammaf(f0);
  // CHECK: f2 = cl::sycl::trunc(f0);
  f2 = truncf(f0);

  // Double Precision Mathematical Functions
  // CHECK: f2 = cl::sycl::acos(f0);
  f2 = acos(f0);
  // CHECK: f2 = cl::sycl::acosh(f0);
  f2 = acosh(f0);
  // CHECK: f2 = cl::sycl::asin(f0);
  f2 = asin(f0);
  // CHECK: f2 = cl::sycl::asinh(f0);
  f2 = asinh(f0);
  // CHECK: f2 = cl::sycl::atan2(f0, f1);
  f2 = atan2(f0, f1);
  // CHECK: f2 = cl::sycl::atan(f0);
  f2 = atan(f0);
  // CHECK: f2 = cl::sycl::atanh(f0);
  f2 = atanh(f0);
  // CHECK: f2 = cl::sycl::cbrt(f0);
  f2 = cbrt(f0);
  // CHECK: f2 = cl::sycl::ceil(f0);
  f2 = ceil(f0);
  // CHECK: f2 = cl::sycl::copysign(f0, f1);
  f2 = copysign(f0, f1);
  // CHECK: f2 = cl::sycl::cos(f0);
  f2 = cos(f0);
  // CHECK: f2 = cl::sycl::cosh(f0);
  f2 = cosh(f0);
  // CHECK: f2 = cl::sycl::cospi(f0);
  f2 = cospi(f0);
  // CHECK: f2 = cl::sycl::erfc(f0);
  f2 = erfc(f0);
  // CHECK: f2 = cl::sycl::erf(f0);
  f2 = erf(f0);
  // CHECK: f2 = cl::sycl::exp10(f0);
  f2 = exp10(f0);
  // CHECK: f2 = cl::sycl::exp2(f0);
  f2 = exp2(f0);
  // CHECK: f2 = cl::sycl::exp(f0);
  f2 = exp(f0);
  // CHECK: f2 = cl::sycl::expm1(f0);
  f2 = expm1(f0);
  // CHECK: f2 = cl::sycl::fabs(f0);
  f2 = fabs(f0);
  // CHECK: f2 = cl::sycl::fdim(f0, f1);
  f2 = fdim(f0, f1);
  // CHECK: f2 = cl::sycl::floor(f0);
  f2 = floor(f0);
  // CHECK: f2 = cl::sycl::fma(f0, f1, f2);
  f2 = fma(f0, f1, f2);
  // CHECK: f2 = cl::sycl::fmax(f0, f1);
  f2 = fmax(f0, f1);
  // CHECK: f2 = cl::sycl::fmin(f0, f1);
  f2 = fmin(f0, f1);
  // CHECK: f2 = cl::sycl::fmod(f0, f1);
  f2 = fmod(f0, f1);
  // CHECK: f2 = cl::sycl::frexp(f0, &i);
  f2 = frexp(f0, &i);
  // CHECK: f2 = cl::sycl::hypot(f0, f1);
  f2 = hypot(f0, f1);
  // CHECK: f2 = cl::sycl::ilogb(f0);
  f2 = ilogb(f0);
  // CHECK: f2 = cl::sycl::ldexp(f0, i);
  f2 = ldexp(f0, i);
  // CHECK: f2 = cl::sycl::lgamma(f0);
  f2 = lgamma(f0);
  // CHECK: f2 = cl::sycl::rint(f0);
  f2 = llrint(f0);
  // CHECK: f2 = cl::sycl::round(f0);
  f2 = llround(f0);
  // CHECK: f2 = cl::sycl::log10(f0);
  f2 = log10(f0);
  // CHECK: f2 = cl::sycl::log1p(f0);
  f2 = log1p(f0);
  // CHECK: f2 = cl::sycl::log2(f0);
  f2 = log2(f0);
  // CHECK: f2 = cl::sycl::logb(f0);
  f2 = logb(f0);
  // CHECK: f2 = cl::sycl::rint(f0);
  f2 = lrint(f0);
  // CHECK: f2 = cl::sycl::round(f0);
  f2 = lround(f0);
  // CHECK: f2 = cl::sycl::modf(f0, &f1);
  f2 = modf(f0, &f1);
  // CHECK: f2 = cl::sycl::nan(0);
  f2 = nan("");
  // CHECK: f2 = cl::sycl::pow(f0, f1);
  f2 = pow(f0, f1);
  // CHECK: f2 = cl::sycl::remainder(f0, f1);
  f2 = remainder(f0, f1);
  // CHECK: f2 = cl::sycl::remquo(f0, f1, &i);
  f2 = remquo(f0, f1, &i);
  // CHECK: f2 = cl::sycl::rint(f0);
  f2 = rint(f0);
  // CHECK: f2 = cl::sycl::round(f0);
  f2 = round(f0);
  // CHECK: f2 = cl::sycl::rsqrt(f0);
  f2 = rsqrt(f0);
  // CHECK: *(&f1) = cl::sycl::sincos(f0, &f2);
  sincosf(f0, &f1, &f2);
  // CHECK: f2 = cl::sycl::sin(f0);
  f2 = sin(f0);
  // CHECK: f2 = cl::sycl::sinh(f0);
  f2 = sinh(f0);
  // CHECK: f2 = cl::sycl::sinpi(f0);
  f2 = sinpi(f0);
  // CHECK: f2 = cl::sycl::sqrt(f0);
  f2 = sqrt(f0);
  // CHECK: f2 = cl::sycl::tan(f0);
  f2 = tan(f0);
  // CHECK: f2 = cl::sycl::tanh(f0);
  f2 = tanh(f0);
  // CHECK: f2 = cl::sycl::tgamma(f0);
  f2 = tgamma(f0);
  // CHECK: f2 = cl::sycl::trunc(f0);
  f2 = trunc(f0);

  // CHECK: f0 = cl::sycl::cos(f0);
  f0 = __cosf(f0);
  // CHECK: f0 = cl::sycl::exp10(f0);
  f0 = __exp10f(f0);
  // CHECK: f0 = cl::sycl::exp(f0);
  f0 = __expf(f0);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 + f1;
  f2 = __fadd_rd(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 + f1;
  f2 = __fadd_rn(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 + f1;
  f2 = __fadd_ru(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 + f1;
  f2 = __fadd_rz(f0, f1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 / f1;
  f2 = __fdiv_rd(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 / f1;
  f2 = __fdiv_rn(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 / f1;
  f2 = __fdiv_ru(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 / f1;
  f2 = __fdiv_rz(f0, f1);

  // CHECK: f2 = cl::sycl::native::divide(f0, f1);
  f2 = __fdividef(f0, f1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = cl::sycl::fma(f0, f1, f2);
  f2 = __fmaf_rd(f0, f1, f2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = cl::sycl::fma(f0, f1, f2);
  f2 = __fmaf_rn(f0, f1, f2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = cl::sycl::fma(f0, f1, f2);
  f2 = __fmaf_ru(f0, f1, f2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = cl::sycl::fma(f0, f1, f2);
  f2 = __fmaf_rz(f0, f1, f2);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK: f2 = f0 * f1;
  f2 = __fmul_rd(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK: f2 = f0 * f1;
  f2 = __fmul_rn(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK: f2 = f0 * f1;
  f2 = __fmul_ru(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK: f2 = f0 * f1;
  f2 = __fmul_rz(f0, f1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::recip(f0);
  f1 = __frcp_rd(f0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::recip(f0);
  f1 = __frcp_rn(f0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::recip(f0);
  f1 = __drcp_ru(f0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::recip(f0);
  f1 = __drcp_rz(f0);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-z :]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f0 = cl::sycl::sqrt(f0);
  f0 = __fsqrt_rd(f0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::sqrt(f1);
  f1 = __fsqrt_rn(f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f0 = cl::sycl::sqrt(f0);
  f0 = __fsqrt_ru(f0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::sqrt(f1);
  f1 = __fsqrt_rz(f1);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 - f1;
  f2 = __fsub_rd(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 - f1;
  f2 = __fsub_rn(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 - f1;
  f2 = __fsub_ru(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 - f1;
  f2 = __fsub_rz(f0, f1);

  // CHECK: f1 = cl::sycl::log10(f1);
  f1 = __log10f(f1);
  // CHECK: f1 = cl::sycl::log2(f1);
  f1 = __log2f(f1);
  // CHECK: f1 = cl::sycl::log(f1);
  f1 = __logf(f1);
  // CHECK: f2 = cl::sycl::pow(f0, f1);
  f2 = __powf(f0, f1);
  // CHECK: *(&f1) = cl::sycl::sincos(f0, &f2);
  __sincosf(f0, &f1, &f2);
  // CHECK: f1 = cl::sycl::sin(f1);
  f1 = __sinf(f1);
  // CHECK: f1 = cl::sycl::tan(f1);
  f1 = __tanf(f1);

  // CHECK: f0 = cl::sycl::fmin(f0, f1);
  f0 = fminf(f0, f1);
  // CHECK: f2 = cl::sycl::fmax(f0, f1);
  f2 = fmaxf(f0, f1);
  // CHECK: f1 = cl::sycl::floor(f1);
  f1 = floorf(f1);
  // CHECK: f2 = cl::sycl::ceil(f2);
  f2 = ceilf(f2);
  // CHECK: f2 = cl::sycl::fma(f0, f1, f2);
  f2 = fmaf(f0, f1, f2);
  // CHECK: f2 = cl::sycl::nan(0);
  f2 = nanf("NaN");

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = cl::sycl::rsqrt(f2);
  f2 = __frsqrt_rn(f2);
}

__global__ void kernelFuncTypecasts() {
  short s, s_1;
  ushort us;
  int i, i_1;
  uint ui, ui_1;
  long l;
  unsigned long ul;
  long long ll;
  unsigned long long ull;

  __half h;
  __half2 h2;
  float f;
  float2 f2;
  double d;
  double2 d2;

  // CHECK: h2 = syclct::float22half2_rn(f2);
  h2 = __float22half2_rn(f2);

  // CHECK: h = syclct::float2half(f);
  h = __float2half(f);

  // CHECK: h2 = syclct::float2half2_rn(f);
  h2 = __float2half2_rn(f);

  // CHECK: h = syclct::float2half_rd(f);
  h = __float2half_rd(f);

  // CHECK: syclct::float2half_rn(f);
  __float2half_rn(f);

  // CHECK: h = syclct::float2half_ru(f);
  h = __float2half_ru(f);

  // CHECK: h = syclct::float2half_rz(f);
  h = __float2half_rz(f);

  // CHECK: h2 = syclct::floats2half2_rn(f, f);
  h2 = __floats2half2_rn(f, f);

  // CHECK; f2 = syclct::half22float2(h2);
  f2 = __half22float2(h2);

  // CHECK: f = syclct::half2float(h);
  f = __half2float(h);

  // CHECK: h2 = syclct::half2half2(h);
  h2 = __half2half2(h);

  // CHECK: i = syclct::half2int_rd(h);
  i = __half2int_rd(h);

  // CHECK: i = syclct::half2int_rn(h);
  i = __half2int_rn(h);

  // CHECK: i = syclct::half2int_ru(h);
  i = __half2int_ru(h);

  // CHECK: i = syclct::half2int_rz(h);
  i = __half2int_rz(h);

  // CHECK: ll = syclct::half2ll_rd(h);
  ll = __half2ll_rd(h);

  // CHECK: ll = syclct::half2ll_rn(h);
  ll = __half2ll_rn(h);

  // CHECK: ll = syclct::half2ll_ru(h);
  ll = __half2ll_ru(h);

  // CHECK: ll = syclct::half2ll_rz(h);
  ll = __half2ll_rz(h);

  // CHECK: s = syclct::half2short_rd(h);
  s = __half2short_rd(h);

  // CHECK: s = syclct::half2short_rn(h);
  s = __half2short_rn(h);

  // CHECK: s = syclct::half2short_ru(h);
  s = __half2short_ru(h);

  // CHECK: s = syclct::half2short_rz(h);
  s = __half2short_rz(h);

  // CHECK: ui = syclct::half2uint_rd(h);
  ui = __half2uint_rd(h);

  // CHECK: ui = syclct::half2uint_rn(h);
  ui = __half2uint_rn(h);

  // CHECK: ui = syclct::half2uint_ru(h);
  ui = __half2uint_ru(h);

  // CHECK: ui = syclct::half2uint_rz(h);
  ui = __half2uint_rz(h);

  // CHECK: ull = syclct::half2ull_rd(h);
  ull = __half2ull_rd(h);

  // CHECK: ull = syclct::half2ull_rn(h);
  ull = __half2ull_rn(h);

  // CHECK: ull = syclct::half2ull_ru(h);
  ull = __half2ull_ru(h);

  // CHECK: ull = syclct::half2ull_rz(h);
  ull = __half2ull_rz(h);

  // CHECK: us = syclct::half2ushort_rd(h);
  us = __half2ushort_rd(h);

  // CHECK: us = syclct::half2ushort_rn(h);
  us = __half2ushort_rn(h);

  // CHECK: us = syclct::half2ushort_ru(h);
  us = __half2ushort_ru(h);

  // CHECK: us = syclct::half2ushort_rz(h);
  us = __half2ushort_rz(h);

  // CHECK: s = syclct::half_as_short(h);
  s = __half_as_short(h);

  // CHECK: us = syclct::half_as_ushort(h);
  us = __half_as_ushort(h);

  // CHECK: h2 = syclct::halves2half2(h, h);
  h2 = __halves2half2(h, h);

  // CHECK: f = syclct::high2float(h2);
  f = __high2float(h2);

  // CHECK: h = syclct::high2half(h2);
  h = __high2half(h2);

  // CHECK: h2 = syclct::high2half2(h2);
  h2 = __high2half2(h2);

  // CHECK: h2 = syclct::highs2half2(h2, h2);
  h2 = __highs2half2(h2, h2);

  // CHECK: h = syclct::int2half_rd(i);
  h = __int2half_rd(i);

  // CHECK: h = syclct::int2half_rn(i);
  h = __int2half_rn(i);

  // CHECK: h = syclct::int2half_ru(i);
  h = __int2half_ru(i);

  // CHECK: h = syclct::int2half_rz(i);
  h = __int2half_rz(i);

  // CHECK: h = syclct::ll2half_rd(ll);
  h = __ll2half_rd(ll);

  // CHECK: h = syclct::ll2half_rn(ll);
  h = __ll2half_rn(ll);

  // CHECK: h = syclct::ll2half_ru(ll);
  h = __ll2half_ru(ll);

  // CHECK: h = syclct::ll2half_rz(ll);
  h = __ll2half_rz(ll);

  // CHECK: f = syclct::low2float(h2);
  f = __low2float(h2);

  // CHECK: h = syclct::low2half(h2);
  h = __low2half(h2);

  // CHECK: h2 = syclct::low2half2(h2);
  h2 = __low2half2(h2);

  // CHECK: h2 = syclct::lowhigh2highlow(h2);
  h2 = __lowhigh2highlow(h2);

  // CHECK: h2 = syclct::lows2half2(h2, h2);
  h2 = __lows2half2(h2, h2);

  // CHECK: h = syclct::short2half_rd(s);
  h = __short2half_rd(s);

  // CHECK: h = syclct::short2half_rn(s);
  h = __short2half_rn(s);

  // CHECK: h = syclct::short2half_ru(s);
  h = __short2half_ru(s);

  // CHECK: h = syclct::short2half_rz(s);
  h = __short2half_rz(s);

  // CHECK: h = syclct::short_as_half(s);
  h = __short_as_half(s);

  // CHECK: h = syclct::uint2half_rd(ui);
  h = __uint2half_rd(ui);

  // CHECK: h = syclct::uint2half_rn(ui);
  h = __uint2half_rn(ui);

  // CHECK: h = syclct::uint2half_ru(ui);
  h = __uint2half_ru(ui);

  // CHECK: h = syclct::uint2half_rz(ui);
  h = __uint2half_rz(ui);

  // CHECK: h = syclct::ull2half_rd(ull);
  h = __ull2half_rd(ull);

  // CHECK: h = syclct::ull2half_rn(ull);
  h = __ull2half_rn(ull);

  // CHECK: h = syclct::ull2half_ru(ull);
  h = __ull2half_ru(ull);

  // CHECK: h = syclct::ull2half_rz(ull);
  h = __ull2half_rz(ull);

  // CHECK: h = syclct::ushort2half_rd(us);
  h = __ushort2half_rd(us);

  // CHECK: h = syclct::ushort2half_rn(us);
  h = __ushort2half_rn(us);

  // CHECK: h = syclct::ushort2half_ru(us);
  h = __ushort2half_ru(us);

  // CHECK: h = syclct::ushort2half_rz(us);
  h = __ushort2half_rz(us);

  // CHECK: h = syclct::ushort_as_half(us);
  h = __ushort_as_half(us);

  // CHECK: f = syclct::double2float_rd(d);
  f = __double2float_rd(d);

  // CHECK: f = syclct::double2float_rn(d);
  f = __double2float_rn(d);

  // CHECK: f = syclct::double2float_ru(d);
  f = __double2float_ru(d);

  // CHECK: f = syclct::double2float_rz(d);
  f = __double2float_rz(d);

  // CHECK: i = syclct::double2int_rd(d);
  i = __double2int_rd(d);

  // CHECK: i = syclct::double2int_rn(d);
  i = __double2int_rn(d);

  // CHECK: i = syclct::double2int_ru(d);
  i = __double2int_ru(d);

  // CHECK: i = syclct::double2int_rz(d);
  i = __double2int_rz(d);

  // CHECK: ll = syclct::double2ll_rd(d);
  ll = __double2ll_rd(d);

  // CHECK: ll = syclct::double2ll_rn(d);
  ll = __double2ll_rn(d);

  // CHECK: ll = syclct::double2ll_ru(d);
  ll = __double2ll_ru(d);

  // CHECK: ll = syclct::double2ll_rz(d);
  ll = __double2ll_rz(d);

  // CHECK: ui = syclct::double2uint_rd(d);
  ui = __double2uint_rd(d);

  // CHECK: ui = syclct::double2uint_rn(d);
  ui = __double2uint_rn(d);

  // CHECK: ui = syclct::double2uint_ru(d);
  ui = __double2uint_ru(d);

  // CHECK: ui = syclct::double2uint_rz(d);
  ui = __double2uint_rz(d);

  // CHECK: ull = syclct::double2ull_rd(d);
  ull = __double2ull_rd(d);

  // CHECK: ull = syclct::double2ull_rn(d);
  ull = __double2ull_rn(d);

  // CHECK: ull = syclct::double2ull_ru(d);
  ull = __double2ull_ru(d);

  // CHECK: ull = syclct::double2ull_rz(d);
  ull = __double2ull_rz(d);

  // CHECK: ll = syclct::double_as_longlong(d);
  ll = __double_as_longlong(d);

  // CHECK: i = syclct::float2int_rd(f);
  i = __float2int_rd(f);

  // CHECK: i = syclct::float2int_rn(f);
  i = __float2int_rn(f);

  // CHECK: i = syclct::float2int_ru(f);
  i = __float2int_ru(f);

  // CHECK: i = syclct::float2int_rz(f);
  i = __float2int_rz(f);

  // CHECK: ll = syclct::float2ll_rd(f);
  ll = __float2ll_rd(f);

  // CHECK: ll = syclct::float2ll_rn(f);
  ll = __float2ll_rn(f);

  // CHECK: ll = syclct::float2ll_ru(f);
  ll = __float2ll_ru(f);

  // CHECK: ll = syclct::float2ll_rz(f);
  ll = __float2ll_rz(f);

  // CHECK: ui = syclct::float2uint_rd(f);
  ui = __float2uint_rd(f);

  // CHECK: ui = syclct::float2uint_rn(f);
  ui = __float2uint_rn(f);

  // CHECK: ui = syclct::float2uint_ru(f);
  ui = __float2uint_ru(f);

  // CHECK: ui = syclct::float2uint_rz(f);
  ui = __float2uint_rz(f);

  // CHECK: ull = syclct::float2ull_rd(f);
  ull = __float2ull_rd(f);

  // CHECK: ull = syclct::float2ull_rn(f);
  ull = __float2ull_rn(f);

  // CHECK: ull = syclct::float2ull_ru(f);
  ull = __float2ull_ru(f);

  // CHECK: ull = syclct::float2ull_rz(f);
  ull = __float2ull_rz(f);

  // CHECK: i = syclct::float_as_int(f);
  i = __float_as_int(f);

  // CHECK: ui = syclct::float_as_uint(f);
  ui = __float_as_uint(f);

  // CHECK: d = syclct::int2double_rn(i);
  d = __int2double_rn(i);

  // CHECK: d = syclct::int2float_rd(i);
  d = __int2float_rd(i);

  // CHECK: d = syclct::int2float_rn(i);
  d = __int2float_rn(i);

  // CHECK: d = syclct::int2float_ru(i);
  d = __int2float_ru(i);

  // CHECK: d = syclct::int2float_rz(i);
  d = __int2float_rz(i);

  // CHECK: f = syclct::int_as_float(i);
  f = __int_as_float(i);

  // CHECK: d = syclct::ll2double_rd(ll);
  d = __ll2double_rd(ll);

  // CHECK: d = syclct::ll2double_rn(ll);
  d = __ll2double_rn(ll);

  // CHECK: d = syclct::ll2double_ru(ll);
  d = __ll2double_ru(ll);

  // CHECK: d = syclct::ll2double_rz(ll);
  d = __ll2double_rz(ll);

  // CHECK: f = syclct::ll2float_rd(ll);
  f = __ll2float_rd(ll);

  // CHECK: f = syclct::ll2float_rn(ll);
  f = __ll2float_rn(ll);

  // CHECK: f = syclct::ll2float_ru(ll);
  f = __ll2float_ru(ll);

  // CHECK: f = syclct::ll2float_rz(ll);
  f = __ll2float_rz(ll);

  // CHECK: d = syclct::longlong_as_double(ll);
  d = __longlong_as_double(ll);

  // CHECK: d = syclct::uint2double_rn(ui);
  d = __uint2double_rn(ui);

  // CHECK: f = syclct::uint2float_rd(ui);
  f = __uint2float_rd(ui);

  // CHECK: f = syclct::uint2float_rn(ui);
  f = __uint2float_rn(ui);

  // CHECK: f = syclct::uint2float_ru(ui);
  f = __uint2float_ru(ui);

  // CHECK: f = syclct::uint2float_rz(ui);
  f = __uint2float_rz(ui);

  // CHECK: f = syclct::uint_as_float(ui);
  f = __uint_as_float(ui);

  // CHECK: d = syclct::ull2double_rd(ull);
  d = __ull2double_rd(ull);

  // CHECK: d = syclct::ull2double_rn(ull);
  d = __ull2double_rn(ull);

  // CHECK: d = syclct::ull2double_ru(ull);
  d = __ull2double_ru(ull);

  // CHECK: d = syclct::ull2double_rz(ull);
  d = __ull2double_rz(ull);

  // CHECK: f = syclct::ull2float_rd(ull);
  f = __ull2float_rd(ull);

  // CHECK: f = syclct::ull2float_rn(ull);
  f = __ull2float_rn(ull);

  // CHECK: f = syclct::ull2float_ru(ull);
  f = __ull2float_ru(ull);

  // CHECK: f = syclct::ull2float_rz(ull);
  f = __ull2float_rz(ull);
}

void testDouble() {
  const unsigned int NUM = 3;
  const unsigned int bytes = NUM * sizeof(double);

  double *hostArrayDouble = (double *)malloc(bytes);
  memset(hostArrayDouble, 0, bytes);
  const long double pi = std::acos(-1.L);
  *hostArrayDouble = pi;
  *(hostArrayDouble + 1) = pi - 1;

  double *deviceArrayDouble;
  cudaMalloc((double **)&deviceArrayDouble, bytes);

  cudaMemcpy(deviceArrayDouble, hostArrayDouble, bytes, cudaMemcpyHostToDevice);

  kernelFuncDouble<<<1, 1>>>(deviceArrayDouble);

  cudaMemcpy(hostArrayDouble, deviceArrayDouble, bytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceArrayDouble);

  cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
       << *(hostArrayDouble + 2) << endl;
}

void testFloat() {
  const unsigned int NUM = 3;
  const unsigned int bytes = NUM * sizeof(float);

  float *hostArrayFloat = (float *)malloc(bytes);
  memset(hostArrayFloat, 0, bytes);
  const long double pi = std::acos(-1.L);
  *hostArrayFloat = pi;
  *(hostArrayFloat + 1) = pi - 1;

  float *deviceArrayFloat;
  cudaMalloc((float **)&deviceArrayFloat, bytes);

  cudaMemcpy(deviceArrayFloat, hostArrayFloat, bytes, cudaMemcpyHostToDevice);

  kernelFuncFloat<<<1, 1>>>(deviceArrayFloat);

  cudaMemcpy(hostArrayFloat, deviceArrayFloat, bytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceArrayFloat);

  cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
       << *(hostArrayFloat + 2) << endl;
}

__global__ void testUnsupported() {
  int i;
  unsigned u;
  long l;
  long long ll;
  unsigned long long ull;
  half h;
  float f;
  double d;
  half2 h2;
  bool b;

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h = __hadd_sat(h, h);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h = __hfma_sat(h, h, h);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h = __hmul_sat(h, h);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h = __hsub_sat(h, h);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hadd2_sat(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hfma2_sat(h2, h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hmul2_sat(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hsub2_sat(h2, h2);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hequ(h, h);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hgeu(h, h);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hgtu(h, h);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hleu(h, h);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hltu(h, h);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hneu(h, h);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbeq2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbequ2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbge2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbgeu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbgt2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbgtu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hble2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbleu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hblt2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbltu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbne2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  b = __hbneu2(h2, h2);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hequ2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hgeu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hgtu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hleu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hltu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2 = __hneu2(h2, h2);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = cyl_bessel_i0f(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = cyl_bessel_i1f(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = erfcinvf(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = erfcxf(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = erfinvf(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = j0f(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = j1f(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = jnf(i, f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = nearbyintf(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = nextafterf(f, f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = norm3df(f, f, f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = norm4df(f, f, f, f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = normcdff(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = normcdfinvf(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = normf(i, &f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = rcbrtf(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = rhypotf(f, f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = rnorm3df(f, f, f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = rnorm4df(f, f, f, f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = rnormf(i, &f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = scalblnf(f, l);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = scalbnf(f, i);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  sincospif(f, &f, &f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = y0f(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = y1f(f);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = ynf(i, f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = cyl_bessel_i0(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = cyl_bessel_i1(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = erfcinv(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = erfcx(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = erfinv(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = j0(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = j1(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = jn(i, d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = nearbyint(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = nextafter(d, d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = norm(i, &d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = norm3d(d, d, d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = norm4d(d, d, d, d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = normcdf(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = normcdfinv(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = rcbrt(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = rnorm3d(d, d, d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = rnorm4d(d, d, d, d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = rnorm(i, &d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = scalbln(d, l);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = scalbn(d, i);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  sincospi(d, &d, &d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = y0(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = y1(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = yn(i, d);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  f = __saturatef(f);

  // i = __shfl_down_sync(u, h, u, i);
  // i = __shfl_sync(u, h, u, i);
  // i = __shfl_up_sync(u, h, u, i);
  // i = __shfl_xor_sync(u, h, u, i);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  i = __double2hiint(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  i = __double2loint(d);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d = __hiloint2double(i, i);


  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __brev(u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  ull = __brevll(ull);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __byte_perm(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  i = __ffs(i);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  i = __ffsll(ll);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __funnelshift_l(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __funnelshift_lc(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __funnelshift_r(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __funnelshift_rc(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  ll = __mul64hi(ll, ll);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  i = __rhadd(i, i);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __sad(i, i, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __uhadd(u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __umul24(u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  ull = __umul64hi(ull, ull);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __umulhi(u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __urhadd(u, u);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  u = __usad(u, u, u);
}

__global__ void testIntegerFunctions() {
  int i;
  unsigned u;
  long l;
  long long ll;
  unsigned long long ull;

  // CHECK: i = cl::sycl::clz(i);
  // CHECK-NEXT: i = cl::sycl::clz(ll);
  // CHECK-NEXT: i = cl::sycl::hadd(i, i);
  // CHECK-NEXT: i = cl::sycl::mul24(i, i);
  // CHECK-NEXT: i = cl::sycl::mul_hi(i, i);
  // CHECK-NEXT: i = cl::sycl::popcount(u);
  // CHECK-NEXT: i = cl::sycl::popcount(ull);
  i = __clz(i);
  i = __clzll(ll);
  i = __hadd(i, i);
  i = __mul24(i, i);
  i = __mulhi(i, i);
  i = __popc(u);
  i = __popcll(ull);
}

void testTypecasts() {

}

int main() {
  testDouble();
  testFloat();
  testTypecasts();
}
