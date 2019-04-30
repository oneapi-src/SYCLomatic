// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/math-functions.sycl.cpp --match-full-lines %s

#include <cmath>

#include <math_functions.h>

#define DECLARE2(type, prefix) \
  type prefix##_a = 0;         \
  type prefix##_b = 0;
#define DECLARE2F DECLARE2(float, f)
#define DECLARE2D DECLARE2(double, d)
#define DECLARE2U DECLARE2(unsigned, u)
#define DECLARE2L DECLARE2(long, l)
#define DECLARE2I DECLARE2(int, i)

#define DECLARE(type, prefix) \
  type prefix##_a = 0;
#define DECLAREF DECLARE(float, f)
#define DECLARED DECLARE(double, d)
#define DECLAREU DECLARE(unsigned, u)
#define DECLAREL DECLARE(long, l)
#define DECLAREI DECLARE(int, i)

int main() {
  // max
  {
    DECLARE2F
    DECLARE2D
    DECLARE2U
    DECLARE2L
    DECLARE2I

    // CHECK: f_b = cl::sycl::max(f_a, f_b);
    f_b = max(f_a, f_b);

    // CHECK: d_b = cl::sycl::max(d_a, d_b);
    d_b = max(d_a, d_b);

    // CHECK: u_b = cl::sycl::max(u_a, u_b);
    u_b = max(u_a, u_b);

    // CHECK: i_b = cl::sycl::max(i_a, i_b);
    i_b = max(i_a, i_b);

    // TODO: Check more primitive type and vector types
  }

  // abs
  {
    DECLAREI

    // CHECK: i_a = cl::sycl::abs(i_a);
    i_a = abs(i_a);

    // TODO: Check more primitive type and vector types
  }

  // acos
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = acos(f_a);
    f_a = acos(f_a);
    // CHECK: d_a = acos(d_a);
    d_a = acos(d_a);

    // TODO: Check more primitive type and vector types
  }

  // acosh
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = acosh(f_a);
    f_a = acosh(f_a);
    // CHECK: d_a = acosh(d_a);
    d_a = acosh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // asin
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = asin(f_a);
    f_a = asin(f_a);
    // CHECK: d_a = asin(d_a);
    d_a = asin(d_a);

    // TODO: Check more primitive type and vector types
  }

  // asinh
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = asinh(f_a);
    f_a = asinh(f_a);
    // CHECK: d_a = asinh(d_a);
    d_a = asinh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // atan
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = atan(f_a);
    f_a = atan(f_a);
    // CHECK: d_a = atan(d_a);
    d_a = atan(d_a);

    // TODO: Check more primitive type and vector types
  }

  // atanh
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = atanh(f_a);
    f_a = atanh(f_a);
    // CHECK: d_a = atanh(d_a);
    d_a = atanh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // cbrt
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = cbrt(f_a);
    f_a = cbrt(f_a);
    // CHECK: d_a = cbrt(d_a);
    d_a = cbrt(d_a);

    // TODO: Check more primitive type and vector types
  }

  // ceil
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = ceil(f_a);
    f_a = ceil(f_a);
    // CHECK: d_a = ceil(d_a);
    d_a = ceil(d_a);

    // TODO: Check more primitive type and vector types
  }

  // cos
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = cos(f_a);
    f_a = cos(f_a);
    // CHECK: d_a = cos(d_a);
    d_a = cos(d_a);

    // TODO: Check more primitive type and vector types
  }

  // cosh
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = cosh(f_a);
    f_a = cosh(f_a);
    // CHECK: d_a = cosh(d_a);
    d_a = cosh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // erfc
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = erfc(f_a);
    f_a = erfc(f_a);
    // CHECK: d_a = erfc(d_a);
    d_a = erfc(d_a);

    // TODO: Check more primitive type and vector types
  }

  // erf
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = erf(f_a);
    f_a = erf(f_a);
    // CHECK: d_a = erf(d_a);
    d_a = erf(d_a);

    // TODO: Check more primitive type and vector types
  }

  // exp
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = cl::sycl::exp(f_a);
    f_a = exp(f_a);
    // CHECK: d_a = cl::sycl::exp(d_a);
    d_a = exp(d_a);

    // TODO: Check more primitive type and vector types
  }

  // exp2
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = exp2(f_a);
    f_a = exp2(f_a);
    // CHECK: d_a = exp2(d_a);
    d_a = exp2(d_a);

    // TODO: Check more primitive type and vector types
  }

  // exp10
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = exp10(f_a);
    f_a = exp10(f_a);
    // CHECK: d_a = exp10(d_a);
    d_a = exp10(d_a);

    // TODO: Check more primitive type and vector types
  }

  // expm1
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = expm1(f_a);
    f_a = expm1(f_a);
    // CHECK: d_a = expm1(d_a);
    d_a = expm1(d_a);

    // TODO: Check more primitive type and vector types
  }

  // fabs
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = cl::sycl::fabs(f_a);
    f_a = fabs(f_a);
    // CHECK: d_a = cl::sycl::fabs(d_a);
    d_a = fabs(d_a);

    // TODO: Check more primitive type and vector types
  }

  // floor
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = floor(f_a);
    f_a = floor(f_a);
    // CHECK: d_a = floor(d_a);
    d_a = floor(d_a);

    // TODO: Check more primitive type and vector types
  }

  // lgamma
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = lgamma(f_a);
    f_a = lgamma(f_a);
    // CHECK: d_a = lgamma(d_a);
    d_a = lgamma(d_a);

    // TODO: Check more primitive type and vector types
  }

  // log
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = log(f_a);
    f_a = log(f_a);
    // CHECK: d_a = log(d_a);
    d_a = log(d_a);

    // TODO: Check more primitive type and vector types
  }

  // log2
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = log2(f_a);
    f_a = log2(f_a);
    // CHECK: d_a = log2(d_a);
    d_a = log2(d_a);

    // TODO: Check more primitive type and vector types
  }

  // log10
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = log10(f_a);
    f_a = log10(f_a);
    // CHECK: d_a = log10(d_a);
    d_a = log10(d_a);

    // TODO: Check more primitive type and vector types
  }

  // log1p
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = log1p(f_a);
    f_a = log1p(f_a);
    // CHECK: d_a = log1p(d_a);
    d_a = log1p(d_a);

    // TODO: Check more primitive type and vector types
  }

  // logb
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = logb(f_a);
    f_a = logb(f_a);
    // CHECK: d_a = logb(d_a);
    d_a = logb(d_a);

    // TODO: Check more primitive type and vector types
  }

  // rint
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = rint(f_a);
    f_a = rint(f_a);
    // CHECK: d_a = rint(d_a);
    d_a = rint(d_a);

    // TODO: Check more primitive type and vector types
  }

  // round
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = round(f_a);
    f_a = round(f_a);
    // CHECK: d_a = round(d_a);
    d_a = round(d_a);

    // TODO: Check more primitive type and vector types
  }

  // rsqrt
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = rsqrt(f_a);
    f_a = rsqrt(f_a);
    // CHECK: d_a = rsqrt(d_a);
    d_a = rsqrt(d_a);

    // TODO: Check more primitive type and vector types
  }

  // sin
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = sin(f_a);
    f_a = sin(f_a);
    // CHECK: d_a = sin(d_a);
    d_a = sin(d_a);

    // TODO: Check more primitive type and vector types
  }

  // sinh
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = sinh(f_a);
    f_a = sinh(f_a);
    // CHECK: d_a = sinh(d_a);
    d_a = sinh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // sqrt
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = cl::sycl::sqrt(f_a);
    f_a = sqrt(f_a);
    // CHECK: d_a = cl::sycl::sqrt(d_a);
    d_a = sqrt(d_a);

    // TODO: Check more primitive type and vector types
  }

  // tan
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = tan(f_a);
    f_a = tan(f_a);
    // CHECK: d_a = tan(d_a);
    d_a = tan(d_a);

    // TODO: Check more primitive type and vector types
  }

  // tanh
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = tanh(f_a);
    f_a = tanh(f_a);
    // CHECK: d_a = tanh(d_a);
    d_a = tanh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // tgamma
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = tgamma(f_a);
    f_a = tgamma(f_a);
    // CHECK: d_a = tgamma(d_a);
    d_a = tgamma(d_a);

    // TODO: Check more primitive type and vector types
  }

  // trunc
  {
    DECLAREF
    DECLARED

    // CHECK: f_a = trunc(f_a);
    f_a = trunc(f_a);
    // CHECK: d_a = trunc(d_a);
    d_a = trunc(d_a);

    // TODO: Check more primitive type and vector types
  }
}
