// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/math-functions.sycl.cpp --match-full-lines %s

#include <algorithm>

#define DECLARE4(type, prefix) \
  type prefix##_a, prefix##_b, prefix##_c, prefix##_d;
#define DECLARE4F DECLARE4(float, f)
#define DECLARE4D DECLARE4(double, d)
#define DECLARE4U DECLARE4(unsigned, u)
#define DECLARE4I DECLARE4(unsigned, i)

#define DECLARE3(type, prefix) type prefix##_a, prefix##_b, prefix##_c;
#define DECLARE3F DECLARE3(float, f)
#define DECLARE3D DECLARE3(double, d)
#define DECLARE3U DECLARE3(unsigned, u)
#define DECLARE3I DECLARE3(unsigned, i)

#define DECLARE2(type, prefix) type prefix##_a, prefix##_b;
#define DECLARE2F DECLARE2(float, f)
#define DECLARE2D DECLARE2(double, d)
#define DECLARE2U DECLARE2(unsigned, u)
#define DECLARE2I DECLARE2(unsigned, i)

#define DECLAREFUNC3(name) template <typename T> \
T name(T, T, T);
#define DECLAREFUNC2(name) template <typename T> \
T name(T, T);
#define DECLAREFUNC1(name) template <typename T> \
T name(T);

DECLAREFUNC1(abs)
DECLAREFUNC1(degrees)
DECLAREFUNC1(radians)
DECLAREFUNC1(sign)

DECLAREFUNC1(cbrt)
DECLAREFUNC1(ceil)
DECLAREFUNC1(erf)
DECLAREFUNC1(erfc)

DECLAREFUNC1(acos)
DECLAREFUNC1(acosh)
DECLAREFUNC1(acospi)
DECLAREFUNC1(asin)
DECLAREFUNC1(asinh)
DECLAREFUNC1(asinpi)
DECLAREFUNC1(atan)
DECLAREFUNC1(atanh)
DECLAREFUNC1(atanpi)
DECLAREFUNC1(cos)
DECLAREFUNC1(cosh)
DECLAREFUNC1(cospi)
DECLAREFUNC1(sin)
DECLAREFUNC1(sinh)
DECLAREFUNC1(sinpi)
DECLAREFUNC1(tan)
DECLAREFUNC1(tanh)
DECLAREFUNC1(tanpi)

DECLAREFUNC2(max)
DECLAREFUNC2(min)
DECLAREFUNC2(step)

DECLAREFUNC3(clamp)
DECLAREFUNC3(mix)
DECLAREFUNC3(smoothstep)

int main() {
  // max
  {
    DECLARE4F
    DECLARE4D
    DECLARE4U
    DECLARE4I

    // CHECK: f_c = cl::sycl::max(f_a, f_b);
    f_c = max(f_a, f_b);

    // CHECK: d_c = cl::sycl::max(d_a, d_b);
    d_c = max(d_a, d_b);

    // CHECK: u_c = cl::sycl::max(u_a, u_b);
    u_c = max(u_a, u_b);

    // CHECK: i_c = cl::sycl::max(i_a, i_b);
    i_c = max(i_a, i_b);

    // CHECK: i_c = std::max(i_a, i_b);
    i_c = std::max(i_a, i_b);

    // TODO: Check more primitive type and vector types
  }

  // clamp
  {
    DECLARE4F
    DECLARE4D
    DECLARE4U
    DECLARE4I

    // CHECK: f_d = cl::sycl::clamp(f_a, f_b, f_c);
    f_d = clamp(f_a, f_b, f_c);
    // CHECK: d_d = cl::sycl::clamp(d_a, d_b, d_c);
    d_d = clamp(d_a, d_b, d_c);
    // CHECK: u_d = cl::sycl::clamp(u_a, u_b, u_c);
    u_d = clamp(u_a, u_b, u_c);
    // CHECK: i_d = cl::sycl::clamp(i_a, i_b, i_c);
    i_d = clamp(i_a, i_b, i_c);

    // TODO: Check more primitive type and vector types
  }

  // degrees
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::degrees(f_a);
    f_b = degrees(f_a);
    // CHECK: d_b = cl::sycl::degrees(d_a);
    d_b = degrees(d_a);

    // TODO: Check more primitive type and vector types
  }

  // abs
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::abs(f_a);
    f_b = abs(f_a);
    // CHECK: d_b = cl::sycl::abs(d_a);
    d_b = abs(d_a);

    // CHECK: f_b = std::abs(f_a);
    f_b = std::abs(f_a);
    // CHECK: d_b = std::abs(d_a);
    d_b = std::abs(d_a);

    // TODO: Check more primitive type and vector types
  }

  // min
  {
    DECLARE3F
    DECLARE3D

    // CHECK: f_c = cl::sycl::min(f_a, f_b);
    f_c = min(f_a, f_b);
    // CHECK: d_c = cl::sycl::min(d_a, d_b);
    d_c = min(d_a, d_b);

    // CHECK: f_c = std::min(f_a, f_b);
    f_c = std::min(f_a, f_b);
    // CHECK: d_c = std::min(d_a, d_b);
    d_c = std::min(d_a, d_b);

    // TODO: Check more primitive type and vector types
  }

  // mix
  {
    DECLARE4F
    DECLARE4D

    // CHECK: f_d = cl::sycl::mix(f_a, f_b, f_c);
    f_d = mix(f_a, f_b, f_c);
    // CHECK: d_d = cl::sycl::mix(d_a, d_b, d_c);
    d_d = mix(d_a, d_b, d_c);

    // TODO: Check more primitive type and vector types
  }

  // radians
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::radians(f_a);
    f_b = radians(f_a);
    // CHECK: d_b = cl::sycl::radians(d_a);
    d_b = radians(d_a);

    // TODO: Check more primitive type and vector types
  }

  // step
  {
    DECLARE3F
    DECLARE3D

    // CHECK: f_c = cl::sycl::step(f_a, f_b);
    f_c = step(f_a, f_b);
    // CHECK: d_c = cl::sycl::step(d_a, d_b);
    d_c = step(d_a, d_b);

    // TODO: Check more primitive type and vector types
  }

  // smoothstep
  {
    DECLARE4F
    DECLARE4D

    // CHECK: f_d = cl::sycl::smoothstep(f_a, f_b, f_c);
    f_d = smoothstep(f_a, f_b, f_c);
    // CHECK: d_d = cl::sycl::smoothstep(d_a, d_b, d_c);
    d_d = smoothstep(d_a, d_b, d_c);

    // TODO: Check more primitive type and vector types
  }

  // sign
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::sign(f_a);
    f_b = sign(f_a);
    // CHECK: d_b = cl::sycl::sign(d_a);
    d_b = sign(d_a);

    // TODO: Check more primitive type and vector types
  }

  // acos
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::acos(f_a);
    f_b = acos(f_a);
    // CHECK: d_b = cl::sycl::acos(d_a);
    d_b = acos(d_a);

    // TODO: Check more primitive type and vector types
  }

  // acosh
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::acosh(f_a);
    f_b = acosh(f_a);
    // CHECK: d_b = cl::sycl::acosh(d_a);
    d_b = acosh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // acospi
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::acospi(f_a);
    f_b = acospi(f_a);
    // CHECK: d_b = cl::sycl::acospi(d_a);
    d_b = acospi(d_a);

    // TODO: Check more primitive type and vector types
  }

  // asin
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::asin(f_a);
    f_b = asin(f_a);
    // CHECK: d_b = cl::sycl::asin(d_a);
    d_b = asin(d_a);

    // TODO: Check more primitive type and vector types
  }

  // asinh
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::asinh(f_a);
    f_b = asinh(f_a);
    // CHECK: d_b = cl::sycl::asinh(d_a);
    d_b = asinh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // asinpi
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::asinpi(f_a);
    f_b = asinpi(f_a);
    // CHECK: d_b = cl::sycl::asinpi(d_a);
    d_b = asinpi(d_a);

    // TODO: Check more primitive type and vector types
  }

  // atan
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::atan(f_a);
    f_b = atan(f_a);
    // CHECK: d_b = cl::sycl::atan(d_a);
    d_b = atan(d_a);

    // TODO: Check more primitive type and vector types
  }

  // atanh
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::atanh(f_a);
    f_b = atanh(f_a);
    // CHECK: d_b = cl::sycl::atanh(d_a);
    d_b = atanh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // atanpi
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::atanpi(f_a);
    f_b = atanpi(f_a);
    // CHECK: d_b = cl::sycl::atanpi(d_a);
    d_b = atanpi(d_a);

    // TODO: Check more primitive type and vector types
  }

  // cbrt
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::cbrt(f_a);
    f_b = cbrt(f_a);
    // CHECK: d_b = cl::sycl::cbrt(d_a);
    d_b = cbrt(d_a);

    // TODO: Check more primitive type and vector types
  }

  // ceil
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::ceil(f_a);
    f_b = ceil(f_a);
    // CHECK: d_b = cl::sycl::ceil(d_a);
    d_b = ceil(d_a);

    // TODO: Check more primitive type and vector types
  }

  // cos
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::cos(f_a);
    f_b = cos(f_a);
    // CHECK: d_b = cl::sycl::cos(d_a);
    d_b = cos(d_a);

    // TODO: Check more primitive type and vector types
  }

  // cosh
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::cosh(f_a);
    f_b = cosh(f_a);
    // CHECK: d_b = cl::sycl::cosh(d_a);
    d_b = cosh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // cospi
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::cospi(f_a);
    f_b = cospi(f_a);
    // CHECK: d_b = cl::sycl::cospi(d_a);
    d_b = cospi(d_a);

    // TODO: Check more primitive type and vector types
  }

  // erfc
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::erfc(f_a);
    f_b = erfc(f_a);
    // CHECK: d_b = cl::sycl::erfc(d_a);
    d_b = erfc(d_a);

    // TODO: Check more primitive type and vector types
  }

  // erf
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::erf(f_a);
    f_b = erf(f_a);
    // CHECK: d_b = cl::sycl::erf(d_a);
    d_b = erf(d_a);

    // TODO: Check more primitive type and vector types
  }

  // exp
  // exp2
  // exp10
  // expm1
  // fabs
  // floor
  // lgamma
  // log
  // log2
  // log10
  // log1p
  // logb
  // rint
  // round
  // rsqrt

  // sin
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::sin(f_a);
    f_b = sin(f_a);
    // CHECK: d_b = cl::sycl::sin(d_a);
    d_b = sin(d_a);

    // TODO: Check more primitive type and vector types
  }

  // sinh
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::sinh(f_a);
    f_b = sinh(f_a);
    // CHECK: d_b = cl::sycl::sinh(d_a);
    d_b = sinh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // sinpi
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::sinpi(f_a);
    f_b = sinpi(f_a);
    // CHECK: d_b = cl::sycl::sinpi(d_a);
    d_b = sinpi(d_a);

    // TODO: Check more primitive type and vector types
  }

  // sqrt

  // tan
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::tan(f_a);
    f_b = tan(f_a);
    // CHECK: d_b = cl::sycl::tan(d_a);
    d_b = tan(d_a);

    // TODO: Check more primitive type and vector types
  }

  // tanh
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::tanh(f_a);
    f_b = tanh(f_a);
    // CHECK: d_b = cl::sycl::tanh(d_a);
    d_b = tanh(d_a);

    // TODO: Check more primitive type and vector types
  }

  // tanpi
  {
    DECLARE2F
    DECLARE2D

    // CHECK: f_b = cl::sycl::tanpi(f_a);
    f_b = tanpi(f_a);
    // CHECK: d_b = cl::sycl::tanpi(d_a);
    d_b = tanpi(d_a);

    // TODO: Check more primitive type and vector types
  }

  // tgamma
  // trunc
}
