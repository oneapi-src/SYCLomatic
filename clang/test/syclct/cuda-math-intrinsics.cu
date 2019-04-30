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

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2_2 = __heq2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2_2 = __hge2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2_2 = __hgt2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2_2 = __hisnan2(h2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2_2 = __hle2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  h2_2 = __hlt2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
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
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
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
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
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
  int i;

  // Double Precision Mathematical Functions
  // CHECK: d2 = cl::sycl::acos(d0);
  d2 = acos(d0);
  // CHECK: d2 = cl::sycl::acosh(d0);
  d2 = acosh(d0);
  // CHECK: d2 = cl::sycl::asin(d0);
  d2 = asin(d0);
  // CHECK: d2 = cl::sycl::asinh(d0);
  d2 = asinh(d0);
  // CHECK: d2 = cl::sycl::atan2(d0, d1);
  d2 = atan2(d0, d1);
  // CHECK: d2 = cl::sycl::atan(d0);
  d2 = atan(d0);
  // CHECK: d2 = cl::sycl::atanh(d0);
  d2 = atanh(d0);
  // CHECK: d2 = cl::sycl::cbrt(d0);
  d2 = cbrt(d0);
  // CHECK: d2 = cl::sycl::ceil(d0);
  d2 = ceil(d0);
  // CHECK: d2 = cl::sycl::copysign(d0, d1);
  d2 = copysign(d0, d1);
  // CHECK: d2 = cl::sycl::cos(d0);
  d2 = cos(d0);
  // CHECK: d2 = cl::sycl::cosh(d0);
  d2 = cosh(d0);
  // CHECK: d2 = cl::sycl::cospi(d0);
  d2 = cospi(d0);
  // CHECK: d2 = cl::sycl::erfc(d0);
  d2 = erfc(d0);
  // CHECK: d2 = cl::sycl::erf(d0);
  d2 = erf(d0);
  // CHECK: d2 = cl::sycl::exp10(d0);
  d2 = exp10(d0);
  // CHECK: d2 = cl::sycl::exp2(d0);
  d2 = exp2(d0);
  // CHECK: d2 = cl::sycl::exp(d0);
  d2 = exp(d0);
  // CHECK: d2 = cl::sycl::expm1(d0);
  d2 = expm1(d0);
  // CHECK: d2 = cl::sycl::fabs(d0);
  d2 = fabs(d0);
  // CHECK: d2 = cl::sycl::fdim(d0, d1);
  d2 = fdim(d0, d1);
  // CHECK: d2 = cl::sycl::floor(d0);
  d2 = floor(d0);
  // CHECK: d2 = cl::sycl::fma(d0, d1, d2);
  d2 = fma(d0, d1, d2);
  // CHECK: d2 = cl::sycl::fmax(d0, d1);
  d2 = fmax(d0, d1);
  // CHECK: d2 = cl::sycl::fmin(d0, d1);
  d2 = fmin(d0, d1);
  // CHECK: d2 = cl::sycl::fmod(d0, d1);
  d2 = fmod(d0, d1);
  // CHECK: d2 = cl::sycl::frexp(d0, cl::sycl::make_ptr<int, cl::sycl::access::address_space::local_space>(&i));
  d2 = frexp(d0, &i);
  // CHECK: d2 = cl::sycl::hypot(d0, d1);
  d2 = hypot(d0, d1);
  // CHECK: d2 = cl::sycl::ilogb(d0);
  d2 = ilogb(d0);
  // CHECK: d2 = cl::sycl::ldexp(d0, i);
  d2 = ldexp(d0, i);
  // CHECK: d2 = cl::sycl::lgamma(d0);
  d2 = lgamma(d0);
  // CHECK: d2 = cl::sycl::rint(d0);
  d2 = llrint(d0);
  // CHECK: d2 = cl::sycl::round(d0);
  d2 = llround(d0);
  // CHECK: d2 = cl::sycl::log10(d0);
  d2 = log10(d0);
  // CHECK: d2 = cl::sycl::log1p(d0);
  d2 = log1p(d0);
  // CHECK: d2 = cl::sycl::log2(d0);
  d2 = log2(d0);
  // CHECK: d2 = cl::sycl::logb(d0);
  d2 = logb(d0);
  // CHECK: d2 = cl::sycl::rint(d0);
  d2 = lrint(d0);
  // CHECK: d2 = cl::sycl::round(d0);
  d2 = lround(d0);
  // CHECK: d2 = cl::sycl::modf(d0, cl::sycl::make_ptr<double, cl::sycl::access::address_space::local_space>(&d1));
  d2 = modf(d0, &d1);
  // CHECK: d2 = cl::sycl::nan(0u);
  d2 = nan("");
  // CHECK: d2 = cl::sycl::pow(d0, d1);
  d2 = pow(d0, d1);
  // CHECK: d2 = cl::sycl::remainder(d0, d1);
  d2 = remainder(d0, d1);
  // CHECK: d2 = cl::sycl::remquo(d0, d1, cl::sycl::make_ptr<int, cl::sycl::access::address_space::local_space>(&i));
  d2 = remquo(d0, d1, &i);
  // CHECK: d2 = cl::sycl::rint(d0);
  d2 = rint(d0);
  // CHECK: d2 = cl::sycl::round(d0);
  d2 = round(d0);
  // CHECK: d2 = cl::sycl::rsqrt(d0);
  d2 = rsqrt(d0);
  // CHECK: d1 = cl::sycl::sincos(d0, cl::sycl::make_ptr<double, cl::sycl::access::address_space::local_space>(&d2));
  sincos(d0, &d1, &d2);
  // CHECK: d2 = cl::sycl::sin(d0);
  d2 = sin(d0);
  // CHECK: d2 = cl::sycl::sinh(d0);
  d2 = sinh(d0);
  // CHECK: d2 = cl::sycl::sinpi(d0);
  d2 = sinpi(d0);
  // CHECK: d2 = cl::sycl::sqrt(d0);
  d2 = sqrt(d0);
  // CHECK: d2 = cl::sycl::tan(d0);
  d2 = tan(d0);
  // CHECK: d2 = cl::sycl::tanh(d0);
  d2 = tanh(d0);
  // CHECK: d2 = cl::sycl::tgamma(d0);
  d2 = tgamma(d0);
  // CHECK: d2 = cl::sycl::trunc(d0);
  d2 = trunc(d0);

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
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d1 = __drcp_rd(d0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d1 = __drcp_rn(d0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
  d1 = __drcp_ru(d0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1004:{{[0-9]+}}: {{[a-zA-z0-9_]+}} is not supported in Sycl
  // CHECK-NEXT: */
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
  // CHECK: d2 = cl::sycl::nan(0u);
  d2 = nan("NaN");

  // CHECK: d0 = cl::sycl::nextafter(d0, d0);
  d0 = nextafterf(d0, d0);
}

__global__ void kernelFuncFloat(float *deviceArrayFloat) {
  float &f0 = *deviceArrayFloat, &f1 = *(deviceArrayFloat + 1), &f2 = *(deviceArrayFloat + 2);
  int i;

  // Single Precision Mathematical Functions
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
  // CHECK: f2 = cl::sycl::frexp(f0, cl::sycl::make_ptr<int, cl::sycl::access::address_space::local_space>(&i));
  f2 = frexpf(f0, &i);
  // CHECK: f2 = cl::sycl::hypot(f0, f1);
  f2 = hypotf(f0, f1);
  // CHECK: f2 = cl::sycl::ilogb(f0);
  f2 = ilogbf(f0);
  // CHECK: i = cl::sycl::isfinite(f0);
  i = isfinite(f0);
  // CHECK: i = cl::sycl::isinf(f0);
  i = isinf(f0);
  // CHECK: i = cl::sycl::isnan(f0);
  i = isnan(f0);
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
  // CHECK: f2 = cl::sycl::modf(f0, cl::sycl::make_ptr<float, cl::sycl::access::address_space::local_space>(&f1));
  f2 = modff(f0, &f1);
  // CHECK: f2 = cl::sycl::nan(0u);
  f2 = nan("");
  // CHECK: f2 = cl::sycl::pow(f0, f1);
  f2 = powf(f0, f1);
  // CHECK: f2 = cl::sycl::remainder(f0, f1);
  f2 = remainderf(f0, f1);
  // CHECK: f2 = cl::sycl::remquo(f0, f1, cl::sycl::make_ptr<int, cl::sycl::access::address_space::local_space>(&i));
  f2 = remquof(f0, f1, &i);
  // CHECK: f2 = cl::sycl::rint(f0);
  f2 = rintf(f0);
  // CHECK: f2 = cl::sycl::round(f0);
  f2 = roundf(f0);
  // CHECK: f2 = cl::sycl::rsqrt(f0);
  f2 = rsqrtf(f0);
  // CHECK: f2 = cl::sycl::signbit(f0);
  f2 = signbit(f0);
  // CHECK: f1 = cl::sycl::sincos(f0, cl::sycl::make_ptr<float, cl::sycl::access::address_space::local_space>(&f2));
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
  // CHECK-NEXT: f1 = cl::sycl::native::recip(f0);
  f1 = __frcp_rd(f0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::native::recip(f0);
  f1 = __frcp_rn(f0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::native::recip(f0);
  f1 = __frcp_ru(f0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = cl::sycl::native::recip(f0);
  f1 = __frcp_rz(f0);

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
  // CHECK: f1 = cl::sycl::sincos(f0, cl::sycl::make_ptr<float, cl::sycl::access::address_space::local_space>(&f2));
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
  // CHECK: f2 = cl::sycl::nan(0u);
  f2 = nanf("NaN");

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = cl::sycl::rsqrt(f2);
  f2 = __frsqrt_rn(f2);

  // CHECK: f0 = cl::sycl::nextafter(f0, f0);
  f0 = nextafterf(f0, f0);
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

  // CHECK: h2 = f2.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>();
  h2 = __float22half2_rn(f2);

  // CHECK: h = cl::sycl::vec<float, 1>{f}.convert<cl::sycl::half, cl::sycl::rounding_mode::automatic>().get_value(0);
  h = __float2half(f);

  // CHECK: h2 = cl::sycl::float2{f,f}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>();
  h2 = __float2half2_rn(f);

  // CHECK: h = cl::sycl::vec<float, 1>{f}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtn>().get_value(0);
  h = __float2half_rd(f);

  // cl::sycl::vec<float, 1>{f}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>().get_value(0);
  __float2half_rn(f);

  // CHECK: h = cl::sycl::vec<float, 1>{f}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtp>().get_value(0);
  h = __float2half_ru(f);

  // CHECK: h = cl::sycl::vec<float, 1>{f}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtz>().get_value(0);
  h = __float2half_rz(f);

  // CHECK: h2 = cl::sycl::float2{f,f}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>();
  h2 = __floats2half2_rn(f, f);

  // CHECK: f2 = h2.convert<float, cl::sycl::rounding_mode::automatic>();
  f2 = __half22float2(h2);

  // CHECK: f = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<float, cl::sycl::rounding_mode::automatic>().get_value(0);
  f = __half2float(h);

  // CHECK: h2 = cl::sycl::half2{h,h};
  h2 = __half2half2(h);

  // CHECK: i = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<int, cl::sycl::rounding_mode::rtn>().get_value(0);
  i = __half2int_rd(h);

  // CHECK: i = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<int, cl::sycl::rounding_mode::rte>().get_value(0);
  i = __half2int_rn(h);

  // CHECK: i = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<int, cl::sycl::rounding_mode::rtp>().get_value(0);
  i = __half2int_ru(h);

  // CHECK: i = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<int, cl::sycl::rounding_mode::rtz>().get_value(0);
  i = __half2int_rz(h);

  // CHECK: ll = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<long long, cl::sycl::rounding_mode::rtn>().get_value(0);
  ll = __half2ll_rd(h);

  // CHECK: ll = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<long long, cl::sycl::rounding_mode::rte>().get_value(0);
  ll = __half2ll_rn(h);

  // CHECK: ll = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<long long, cl::sycl::rounding_mode::rtp>().get_value(0);
  ll = __half2ll_ru(h);

  // CHECK: ll = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<long long, cl::sycl::rounding_mode::rtz>().get_value(0);
  ll = __half2ll_rz(h);

  // CHECK: s = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<short, cl::sycl::rounding_mode::rtn>().get_value(0);
  s = __half2short_rd(h);

  // CHECK: s = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<short, cl::sycl::rounding_mode::rte>().get_value(0);
  s = __half2short_rn(h);

  // CHECK: s = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<short, cl::sycl::rounding_mode::rtp>().get_value(0);
  s = __half2short_ru(h);

  // CHECK: s = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<short, cl::sycl::rounding_mode::rtz>().get_value(0);
  s = __half2short_rz(h);

  // CHECK: ui = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned int, cl::sycl::rounding_mode::rtn>().get_value(0);
  ui = __half2uint_rd(h);

  // CHECK: ui = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned int, cl::sycl::rounding_mode::rte>().get_value(0);
  ui = __half2uint_rn(h);

  // CHECK:ui = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned int, cl::sycl::rounding_mode::rtp>().get_value(0);
  ui = __half2uint_ru(h);

  // CHECK: ui = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned int, cl::sycl::rounding_mode::rtz>().get_value(0);
  ui = __half2uint_rz(h);

  // CHECK: ull = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned long long, cl::sycl::rounding_mode::rtn>().get_value(0);
  ull = __half2ull_rd(h);

  // CHECK: ull = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned long long, cl::sycl::rounding_mode::rte>().get_value(0);
  ull = __half2ull_rn(h);

  // CHECK: ull = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned long long, cl::sycl::rounding_mode::rtp>().get_value(0);
  ull = __half2ull_ru(h);

  // CHECK: ull = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned long long, cl::sycl::rounding_mode::rtz>().get_value(0);
  ull = __half2ull_rz(h);

  // CHECK: us = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned short, cl::sycl::rounding_mode::rtn>().get_value(0);
  us = __half2ushort_rd(h);

  // CHECK: us = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned short, cl::sycl::rounding_mode::rte>().get_value(0);
  us = __half2ushort_rn(h);

  // CHECK: us = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned short, cl::sycl::rounding_mode::rtp>().get_value(0);
  us = __half2ushort_ru(h);

  // CHECK: us = cl::sycl::vec<cl::sycl::half, 1>{h}.convert<unsigned short, cl::sycl::rounding_mode::rtz>().get_value(0);
  us = __half2ushort_rz(h);

  // CHECK: s = syclct::bit_cast<cl::sycl::half, short>(h);
  s = __half_as_short(h);

  // CHECK: us = syclct::bit_cast<cl::sycl::half, unsigned short>(h);
  us = __half_as_ushort(h);

  // CHECK: h2 = cl::sycl::half2{h,h};
  h2 = __halves2half2(h, h);

  // CHECK: f = h2.get_value(0);
  f = __high2float(h2);

  // CHECK: h = h2.get_value(0);
  h = __high2half(h2);

  // CHECK: h2 = cl::sycl::half2{h2.get_value(0), h2.get_value(0)};
  h2 = __high2half2(h2);

  // CHECK: h2 = cl::sycl::half2{h2.get_value(0), h2.get_value(0)};
  h2 = __highs2half2(h2, h2);

  // CHECK: h = cl::sycl::vec<int, 1>{i}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtn>().get_value(0);
  h = __int2half_rd(i);

  // CHECK: h = cl::sycl::vec<int, 1>{i}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>().get_value(0);
  h = __int2half_rn(i);

  // CHECK: h = cl::sycl::vec<int, 1>{i}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtp>().get_value(0);
  h = __int2half_ru(i);

  // CHECK: h = cl::sycl::vec<int, 1>{i}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtz>().get_value(0);
  h = __int2half_rz(i);

  // CHECK: h = cl::sycl::vec<long long, 1>{ll}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtn>().get_value(0);
  h = __ll2half_rd(ll);

  // CHECK: h = cl::sycl::vec<long long, 1>{ll}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>().get_value(0);
  h = __ll2half_rn(ll);

  // CHECK: h = cl::sycl::vec<long long, 1>{ll}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtp>().get_value(0);
  h = __ll2half_ru(ll);

  // CHECK: h = cl::sycl::vec<long long, 1>{ll}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtz>().get_value(0);
  h = __ll2half_rz(ll);

  // CHECK: f = h2.get_value(1);
  f = __low2float(h2);

  // CHECK: h = h2.get_value(1);
  h = __low2half(h2);

  // CHECK: h2 = cl::sycl::half2{h2.get_value(1), h2.get_value(1)};
  h2 = __low2half2(h2);

  // CHECK: h2 = cl::sycl::half2{h2.get_value(1), h2.get_value(0)};
  h2 = __lowhigh2highlow(h2);

  // CHECK: h2 = cl::sycl::half2{h2.get_value(1), h2.get_value(1)};
  h2 = __lows2half2(h2, h2);

  // CHECK: h = cl::sycl::vec<short, 1>{s}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtn>().get_value(0);
  h = __short2half_rd(s);

  // CHECK: h = cl::sycl::vec<short, 1>{s}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>().get_value(0);
  h = __short2half_rn(s);

  // CHECK: h = cl::sycl::vec<short, 1>{s}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtp>().get_value(0);
  h = __short2half_ru(s);

  // CHECK: h = cl::sycl::vec<short, 1>{s}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtz>().get_value(0);
  h = __short2half_rz(s);

  // CHECK: h = syclct::bit_cast<short, cl::sycl::half>(s);
  h = __short_as_half(s);

  // CHECK: h = cl::sycl::vec<unsigned int, 1>{ui}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtn>().get_value(0);
  h = __uint2half_rd(ui);

  // CHECK: h = cl::sycl::vec<unsigned int, 1>{ui}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>().get_value(0);
  h = __uint2half_rn(ui);

  // CHECK: h = cl::sycl::vec<unsigned int, 1>{ui}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtp>().get_value(0);
  h = __uint2half_ru(ui);

  // CHECK: h = cl::sycl::vec<unsigned int, 1>{ui}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtz>().get_value(0);
  h = __uint2half_rz(ui);

  // CHECK: h = cl::sycl::vec<unsigned long long, 1>{ull}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtn>().get_value(0);
  h = __ull2half_rd(ull);

  // CHECK: h = cl::sycl::vec<unsigned long long, 1>{ull}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>().get_value(0);
  h = __ull2half_rn(ull);

  // CHECK: h = cl::sycl::vec<unsigned long long, 1>{ull}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtp>().get_value(0);
  h = __ull2half_ru(ull);

  // CHECK: h = cl::sycl::vec<unsigned long long, 1>{ull}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtz>().get_value(0);
  h = __ull2half_rz(ull);

  // CHECK: h = cl::sycl::vec<unsigned short, 1>{us}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtn>().get_value(0);
  h = __ushort2half_rd(us);

  // CHECK: h = cl::sycl::vec<unsigned short, 1>{us}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>().get_value(0);
  h = __ushort2half_rn(us);

  // CHECK: h = cl::sycl::vec<unsigned short, 1>{us}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtp>().get_value(0);
  h = __ushort2half_ru(us);

  // CHECK: h = cl::sycl::vec<unsigned short, 1>{us}.convert<cl::sycl::half, cl::sycl::rounding_mode::rtz>().get_value(0);
  h = __ushort2half_rz(us);

  // CHECK: h = syclct::bit_cast<unsigned short, cl::sycl::half>(us);
  h = __ushort_as_half(us);

  // CHECK: f = cl::sycl::vec<double, 1>{d}.convert<float, cl::sycl::rounding_mode::rtn>().get_value(0);
  f = __double2float_rd(d);

  // CHECK: f = cl::sycl::vec<double, 1>{d}.convert<float, cl::sycl::rounding_mode::rte>().get_value(0);
  f = __double2float_rn(d);

  // CHECK: f = cl::sycl::vec<double, 1>{d}.convert<float, cl::sycl::rounding_mode::rtp>().get_value(0);
  f = __double2float_ru(d);

  // CHECK: f = cl::sycl::vec<double, 1>{d}.convert<float, cl::sycl::rounding_mode::rtz>().get_value(0);
  f = __double2float_rz(d);

  // CHECK: i = cl::sycl::vec<double, 1>{d}.convert<int, cl::sycl::rounding_mode::rtn>().get_value(0);
  i = __double2int_rd(d);

  // CHECK: i = cl::sycl::vec<double, 1>{d}.convert<int, cl::sycl::rounding_mode::rte>().get_value(0);
  i = __double2int_rn(d);

  // CHECK: i = cl::sycl::vec<double, 1>{d}.convert<int, cl::sycl::rounding_mode::rtp>().get_value(0);
  i = __double2int_ru(d);

  // CHECK: i = cl::sycl::vec<double, 1>{d}.convert<int, cl::sycl::rounding_mode::rtz>().get_value(0);
  i = __double2int_rz(d);

  // CHECK: ll = cl::sycl::vec<double, 1>{d}.convert<long long, cl::sycl::rounding_mode::rtn>().get_value(0);
  ll = __double2ll_rd(d);

  // CHECK: ll = cl::sycl::vec<double, 1>{d}.convert<long long, cl::sycl::rounding_mode::rte>().get_value(0);
  ll = __double2ll_rn(d);

  // CHECK: ll = cl::sycl::vec<double, 1>{d}.convert<long long, cl::sycl::rounding_mode::rtp>().get_value(0);
  ll = __double2ll_ru(d);

  // CHECK: ll = cl::sycl::vec<double, 1>{d}.convert<long long, cl::sycl::rounding_mode::rtz>().get_value(0);
  ll = __double2ll_rz(d);

  // CHECK: ui = cl::sycl::vec<double, 1>{d}.convert<unsigned int, cl::sycl::rounding_mode::rtn>().get_value(0);
  ui = __double2uint_rd(d);

  // CHECK:ui = cl::sycl::vec<double, 1>{d}.convert<unsigned int, cl::sycl::rounding_mode::rte>().get_value(0);
  ui = __double2uint_rn(d);

  // CHECK: ui = cl::sycl::vec<double, 1>{d}.convert<unsigned int, cl::sycl::rounding_mode::rtp>().get_value(0);
  ui = __double2uint_ru(d);

  // CHECK: ui = cl::sycl::vec<double, 1>{d}.convert<unsigned int, cl::sycl::rounding_mode::rtz>().get_value(0);
  ui = __double2uint_rz(d);

  // CHECK: ull = cl::sycl::vec<double, 1>{d}.convert<unsigned long long, cl::sycl::rounding_mode::rtn>().get_value(0);
  ull = __double2ull_rd(d);

  // CHECK: ull = cl::sycl::vec<double, 1>{d}.convert<unsigned long long, cl::sycl::rounding_mode::rte>().get_value(0);
  ull = __double2ull_rn(d);

  // CHECK: ull = cl::sycl::vec<double, 1>{d}.convert<unsigned long long, cl::sycl::rounding_mode::rtp>().get_value(0);
  ull = __double2ull_ru(d);

  // CHECK: ull = cl::sycl::vec<double, 1>{d}.convert<unsigned long long, cl::sycl::rounding_mode::rtz>().get_value(0);
  ull = __double2ull_rz(d);

  // CHECK: ll = syclct::bit_cast<double, long long>(d);
  ll = __double_as_longlong(d);

  // CHECK: i = cl::sycl::vec<float, 1>{f}.convert<int, cl::sycl::rounding_mode::rtn>().get_value(0);
  i = __float2int_rd(f);

  // CHECK: i = cl::sycl::vec<float, 1>{f}.convert<int, cl::sycl::rounding_mode::rte>().get_value(0);
  i = __float2int_rn(f);

  // CHECK: i = cl::sycl::vec<float, 1>{f}.convert<int, cl::sycl::rounding_mode::rtp>().get_value(0);
  i = __float2int_ru(f);

  // CHECK: i = cl::sycl::vec<float, 1>{f}.convert<int, cl::sycl::rounding_mode::rtz>().get_value(0);
  i = __float2int_rz(f);

  // CHECK: ll = cl::sycl::vec<float, 1>{f}.convert<long long, cl::sycl::rounding_mode::rtn>().get_value(0);
  ll = __float2ll_rd(f);

  // CHECK: ll = cl::sycl::vec<float, 1>{f}.convert<long long, cl::sycl::rounding_mode::rte>().get_value(0);
  ll = __float2ll_rn(f);

  // CHECK: ll = cl::sycl::vec<float, 1>{f}.convert<long long, cl::sycl::rounding_mode::rtp>().get_value(0);
  ll = __float2ll_ru(f);

  // CHECK: ll = cl::sycl::vec<float, 1>{f}.convert<long long, cl::sycl::rounding_mode::rtz>().get_value(0);
  ll = __float2ll_rz(f);

  // CHECK: ui = cl::sycl::vec<float, 1>{f}.convert<unsigned int, cl::sycl::rounding_mode::rtn>().get_value(0);
  ui = __float2uint_rd(f);

  // CHECK: ui = cl::sycl::vec<float, 1>{f}.convert<unsigned int, cl::sycl::rounding_mode::rte>().get_value(0);
  ui = __float2uint_rn(f);

  // CHECK: ui = cl::sycl::vec<float, 1>{f}.convert<unsigned int, cl::sycl::rounding_mode::rtp>().get_value(0);
  ui = __float2uint_ru(f);

  // CHECK: ui = cl::sycl::vec<float, 1>{f}.convert<unsigned int, cl::sycl::rounding_mode::rtz>().get_value(0);
  ui = __float2uint_rz(f);

  // CHECK: ull = cl::sycl::vec<float, 1>{f}.convert<unsigned long long, cl::sycl::rounding_mode::rtn>().get_value(0);
  ull = __float2ull_rd(f);

  // CHECK: ull = cl::sycl::vec<float, 1>{f}.convert<unsigned long long, cl::sycl::rounding_mode::rte>().get_value(0);
  ull = __float2ull_rn(f);

  // CHECK: ull = cl::sycl::vec<float, 1>{f}.convert<unsigned long long, cl::sycl::rounding_mode::rtp>().get_value(0);
  ull = __float2ull_ru(f);

  // CHECK: ull = cl::sycl::vec<float, 1>{f}.convert<unsigned long long, cl::sycl::rounding_mode::rtz>().get_value(0);
  ull = __float2ull_rz(f);

  // CHECK: i = syclct::bit_cast<float, int>(f);
  i = __float_as_int(f);

  // CHECK: ui = syclct::bit_cast<float, unsigned int>(f);
  ui = __float_as_uint(f);

  // CHECK: d = cl::sycl::vec<int, 1>{i}.convert<double, cl::sycl::rounding_mode::rte>().get_value(0);
  d = __int2double_rn(i);

  // CHECK: d = cl::sycl::vec<int, 1>{i}.convert<float, cl::sycl::rounding_mode::rtn>().get_value(0);
  d = __int2float_rd(i);

  // CHECK: d = cl::sycl::vec<int, 1>{i}.convert<float, cl::sycl::rounding_mode::rte>().get_value(0);
  d = __int2float_rn(i);

  // CHECK: d = cl::sycl::vec<int, 1>{i}.convert<float, cl::sycl::rounding_mode::rtp>().get_value(0);
  d = __int2float_ru(i);

  // CHECK: d = cl::sycl::vec<int, 1>{i}.convert<float, cl::sycl::rounding_mode::rtz>().get_value(0);
  d = __int2float_rz(i);

  // CHECK: f = syclct::bit_cast<int, float>(i);
  f = __int_as_float(i);

  // CHECK: d = cl::sycl::vec<long long, 1>{ll}.convert<double, cl::sycl::rounding_mode::rtn>().get_value(0);
  d = __ll2double_rd(ll);

  // CHECK: d = cl::sycl::vec<long long, 1>{ll}.convert<double, cl::sycl::rounding_mode::rte>().get_value(0);
  d = __ll2double_rn(ll);

  // CHECK: d = cl::sycl::vec<long long, 1>{ll}.convert<double, cl::sycl::rounding_mode::rtp>().get_value(0);
  d = __ll2double_ru(ll);

  // CHECK: d = cl::sycl::vec<long long, 1>{ll}.convert<double, cl::sycl::rounding_mode::rtz>().get_value(0);
  d = __ll2double_rz(ll);

  // CHECK: f = cl::sycl::vec<long long, 1>{ll}.convert<float, cl::sycl::rounding_mode::rtn>().get_value(0);
  f = __ll2float_rd(ll);

  // CHECK: f = cl::sycl::vec<long long, 1>{ll}.convert<float, cl::sycl::rounding_mode::rte>().get_value(0);
  f = __ll2float_rn(ll);

  // CHECK: f = cl::sycl::vec<long long, 1>{ll}.convert<float, cl::sycl::rounding_mode::rtp>().get_value(0);
  f = __ll2float_ru(ll);

  // CHECK: f = cl::sycl::vec<long long, 1>{ll}.convert<float, cl::sycl::rounding_mode::rtz>().get_value(0);
  f = __ll2float_rz(ll);

  // CHECK: d = syclct::bit_cast<long long, double>(ll);
  d = __longlong_as_double(ll);

  // CHECK: d = cl::sycl::vec<unsigned int, 1>{ui}.convert<double, cl::sycl::rounding_mode::rte>().get_value(0);
  d = __uint2double_rn(ui);

  // CHECK: f = cl::sycl::vec<unsigned int, 1>{ui}.convert<float, cl::sycl::rounding_mode::rtn>().get_value(0);
  f = __uint2float_rd(ui);

  // CHECK: f = cl::sycl::vec<unsigned int, 1>{ui}.convert<float, cl::sycl::rounding_mode::rte>().get_value(0);
  f = __uint2float_rn(ui);

  // CHECK: f = cl::sycl::vec<unsigned int, 1>{ui}.convert<float, cl::sycl::rounding_mode::rtp>().get_value(0);
  f = __uint2float_ru(ui);

  // CHECK: f = cl::sycl::vec<unsigned int, 1>{ui}.convert<float, cl::sycl::rounding_mode::rtz>().get_value(0);
  f = __uint2float_rz(ui);

  // CHECK: f = syclct::bit_cast<unsigned int, float>(ui);
  f = __uint_as_float(ui);

  // CHECK: d = cl::sycl::vec<unsigned long long, 1>{ull}.convert<double, cl::sycl::rounding_mode::rtn>().get_value(0);
  d = __ull2double_rd(ull);

  // CHECK: d = cl::sycl::vec<unsigned long long, 1>{ull}.convert<double, cl::sycl::rounding_mode::rte>().get_value(0);
  d = __ull2double_rn(ull);

  // CHECK: d = cl::sycl::vec<unsigned long long, 1>{ull}.convert<double, cl::sycl::rounding_mode::rtp>().get_value(0);
  d = __ull2double_ru(ull);

  // CHECK: d = cl::sycl::vec<unsigned long long, 1>{ull}.convert<double, cl::sycl::rounding_mode::rtz>().get_value(0);
  d = __ull2double_rz(ull);

  // CHECK: f = cl::sycl::vec<unsigned long long, 1>{ull}.convert<float, cl::sycl::rounding_mode::rtn>().get_value(0);
  f = __ull2float_rd(ull);

  // CHECK: f = cl::sycl::vec<unsigned long long, 1>{ull}.convert<float, cl::sycl::rounding_mode::rte>().get_value(0);
  f = __ull2float_rn(ull);

  // CHECK: f = cl::sycl::vec<unsigned long long, 1>{ull}.convert<float, cl::sycl::rounding_mode::rtp>().get_value(0);
  f = __ull2float_ru(ull);

  // CHECK: f = cl::sycl::vec<unsigned long long, 1>{ull}.convert<float, cl::sycl::rounding_mode::rtz>().get_value(0);
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

__global__ void testSimulation() {
  float f;
  double d;

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1017:{{[0-9]+}}: nearbyintf is simulated by cl::sycl::floor in DPC++. Please check the potential precision and/or performance issues of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = cl::sycl::floor(f + 0.5);
  f = nearbyintf(f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1017:{{[0-9]+}}: nearbyint is simulated by cl::sycl::floor in DPC++. Please check the potential precision and/or performance issues of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d = cl::sycl::floor(d + 0.5);
  d = nearbyint(d);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1017:{{[0-9]+}}: rhypotf is simulated by cl::sycl::hypot in DPC++. Please check the potential precision and/or performance issues of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = 1 / cl::sycl::hypot(f, f);
  f = rhypotf(f, f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1017:{{[0-9]+}}: sincospif is simulated by cl::sycl::sincos in DPC++. Please check the potential precision and/or performance issues of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = cl::sycl::sincos(f * SYCLCT_PI_F, cl::sycl::make_ptr<float, cl::sycl::access::address_space::local_space>(&f));
  sincospif(f, &f, &f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1017:{{[0-9]+}}: sincospi is simulated by cl::sycl::sincos in DPC++. Please check the potential precision and/or performance issues of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d = cl::sycl::sincos(d * SYCLCT_PI, cl::sycl::make_ptr<double, cl::sycl::access::address_space::local_space>(&d));
  sincospi(d, &d, &d);
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
