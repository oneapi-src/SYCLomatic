// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none -out-root %T/math_functions_test %s --cuda-include-path="%cuda-path/include" --optimize-migration -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/math_functions_test/math_functions_test.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <cmath>

// CHECK: int test_signbit(float x) { return signbit(x); }
int test_signbit(float x) { return signbit(x); }

// CHECK: int test_signbit(double x) { return signbit(x); }
int test_signbit(double x) { return signbit(x); }

// CHECK: int test_signbit(long double x) { return signbit((double)x); }
int test_signbit(long double x) { return signbit((double)x); }

// CHECK: int test_isfinite(float x) { return isfinite(x); }
int test_isfinite(float x) { return isfinite(x); }

// CHECK: int test_isfinite(double x) { return isfinite(x); }
int test_isfinite(double x) { return isfinite(x); }

// CHECK: int test_isfinite(long double x) { return isfinite((double)x); }
int test_isfinite(long double x) { return isfinite((double)x); }

// CHECK: int test_isnan(float x) { return isnan(x); }
int test_isnan(float x) { return isnan(x); }

// CHECK: int test_isnan(double x) { return isnan(x); }
int test_isnan(double x) { return isnan(x); }

// CHECK: int test_isnan(long double x) { return isnan((double)x); }
int test_isnan(long double x) { return isnan((double)x); }

// CHECK: int test_isinf(float x) { return isinf(x); }
int test_isinf(float x) { return isinf(x); }

// CHECK: int test_isinf(double x) { return isinf(x); }
int test_isinf(double x) { return isinf(x); }

// CHECK: int test_isinf(long double x) { return isinf((double)x); }
int test_isinf(long double x) { return isinf((double)x); }

// CHECK: long long int test_abs(long long int a) { return abs(a); }
long long int test_abs(long long int a) { return abs(a); }

// CHECK: long int test_abs(long int in) { return abs(in); }
long int test_abs(long int in) { return abs(in); }

// CHECK: float test_abs(float in) { return abs(in); }
float test_abs(float in) { return abs(in); }

// CHECK: double test_abs(double in) { return abs(in); }
double test_abs(double in) { return abs(in); }

// CHECK: float test_fabs(float in) { return fabs(in); }
float test_fabs(float in) { return fabs(in); }

// CHECK: float test_ceil(float in) { return ceil(in); }
float test_ceil(float in) { return ceil(in); }

// CHECK: float test_floor(float in) { return floor(in); }
float test_floor(float in) { return floor(in); }

// CHECK: float test_sqrt(float in) { return sqrt(in); }
float test_sqrt(float in) { return sqrt(in); }

// CHECK: float test_pow(float a, float b) { return pow(a, b); }
float test_pow(float a, float b) { return pow(a, b); }

// CHECK: float test_pow(float a, int b) { return sycl::pown<double>(a, b); }
float test_pow(float a, int b) { return pow(a, b); }

// CHECK: double test_pow(double a, int b) { return sycl::pown<double>(a, b); }
double test_pow(double a, int b) { return pow(a, b); }

// CHECK: float test_powif(float a, int b) { return powif(a, b); }
float test_powif(float a, int b) { return powif(a, b); }

// CHECK: double test_powi(double a, int b) { return powi(a, b); }
double test_powi(double a, int b) { return powi(a, b); }

// CHECK: float test_log(float in) { return log(in); }
float test_log(float in) { return log(in); }

// CHECK: float test_log10(float in) { return log10(in); }
float test_log10(float in) { return log10(in); }

// CHECK: float test_fmod(float a, float b) { return fmod(a, b); }
float test_fmod(float a, float b) { return fmod(a, b); }

// CHECK: float test_modf(float a, float *b) { return modf(a, b); }
float test_modf(float a, float *b) { return modf(a, b); }

// CHECK: float test_exp(float in) { return exp(in); }
float test_exp(float in) { return exp(in); }

// CHECK: float test_frexp(float a, int *b) { return frexp(a, b); }
float test_frexp(float a, int *b) { return frexp(a, b); }

// CHECK: float test_ldexp(float a, int b) { return ldexp(a, b); }
float test_ldexp(float a, int b) { return ldexp(a, b); }

// CHECK: float test_asin(float in) { return asin(in); }
float test_asin(float in) { return asin(in); }

// CHECK: float test_sin(float in) { return sin(in); }
float test_sin(float in) { return sin(in); }

// CHECK: float test_sinh(float in) { return sinh(in); }
float test_sinh(float in) { return sinh(in); }

// CHECK: float test_acos(float in) { return acos(in); }
float test_acos(float in) { return acos(in); }

// CHECK: float test_cos(float in) { return cos(in); }
float test_cos(float in) { return cos(in); }

// CHECK: float test_cosh(float in) { return cosh(in); }
float test_cosh(float in) { return cosh(in); }

// CHECK: float test_atan(float in) { return atan(in); }
float test_atan(float in) { return atan(in); }

// CHECK: float test_atan2(float a, float b) { return atan2(a, b); }
float test_atan2(float a, float b) { return atan2(a, b); }

// CHECK: float test_tan(float in) { return tan(in); }
float test_tan(float in) { return tan(in); }

// CHECK: float test_tanh(float in) { return tanh(in); }
float test_tanh(float in) { return tanh(in); }

// CHECK: float test_logb(float a) { return logb(a); }
float test_logb(float a) { return logb(a); }

// CHECK: int test_ilogb(float a) { return ilogb(a); }
int test_ilogb(float a) { return ilogb(a); }

// CHECK: float test_scalbn(float a, int b) { return scalbn(a, b); }
float test_scalbn(float a, int b) { return scalbn(a, b); }

// CHECK: float test_scalbln(float a, long int b) { return scalbln(a, b); }
float test_scalbln(float a, long int b) { return scalbln(a, b); }

// CHECK: float test_exp2(float a) { return exp2(a); }
float test_exp2(float a) { return exp2(a); }

// CHECK: float test_expm1(float a) { return expm1(a); }
float test_expm1(float a) { return expm1(a); }

// CHECK: float test_log2(float a) { return log2(a); }
float test_log2(float a) { return log2(a); }

// CHECK: float test_log1p(float a) { return log1p(a); }
float test_log1p(float a) { return log1p(a); }

// CHECK: float test_acosh(float a) { return acosh(a); }
float test_acosh(float a) { return acosh(a); }

// CHECK: float test_asinh(float a) { return asinh(a); }
float test_asinh(float a) { return asinh(a); }

// CHECK: float test_atanh(float a) { return atanh(a); }
float test_atanh(float a) { return atanh(a); }

// CHECK: float test_hypot(float a, float b) { return hypot(a, b); }
float test_hypot(float a, float b) { return hypot(a, b); }

// CHECK: float test_norm3d(float a, float b, float c) { return norm3d(a, b, c); }
float test_norm3d(float a, float b, float c) { return norm3d(a, b, c); }

// CHECK: float test_norm4d(float a, float b, float c, float d) {
// CHECK:   return norm4d(a, b, c, d);
// CHECK: }
float test_norm4d(float a, float b, float c, float d) {
  return norm4d(a, b, c, d);
}

// CHECK: float test_cbrt(float a) { return cbrt(a); }
float test_cbrt(float a) { return cbrt(a); }

// CHECK: float test_erf(float a) { return erf(a); }
float test_erf(float a) { return erf(a); }

// CHECK: float test_erfc(float a) { return erfc(a); }
float test_erfc(float a) { return erfc(a); }

// CHECK: float test_lgamma(float a) { return lgamma(a); }
float test_lgamma(float a) { return lgamma(a); }

// CHECK: float test_tgamma(float a) { return tgamma(a); }
float test_tgamma(float a) { return tgamma(a); }

// CHECK: float test_copysign(float a, float b) { return copysign(a, b); }
float test_copysign(float a, float b) { return copysign(a, b); }

// CHECK: float test_nextafter(float a, float b) { return nextafter(a, b); }
float test_nextafter(float a, float b) { return nextafter(a, b); }

// CHECK: float test_remainder(float a, float b) { return remainder(a, b); }
float test_remainder(float a, float b) { return remainder(a, b); }

// CHECK: float test_remquo(float a, float b, int *quo) { return remquo(a, b, quo); }
float test_remquo(float a, float b, int *quo) { return remquo(a, b, quo); }

// CHECK: float test_round(float a) { return round(a); }
float test_round(float a) { return round(a); }

// CHECK: long int test_lround(float a) { return lround(a); }
long int test_lround(float a) { return lround(a); }

// CHECK: long long int test_llround(float a) { return llround(a); }
long long int test_llround(float a) { return llround(a); }

// CHECK: float test_trunc(float a) { return trunc(a); }
float test_trunc(float a) { return trunc(a); }

// CHECK: float test_rint(float a) { return rint(a); }
float test_rint(float a) { return rint(a); }

// CHECK: long int test_lrint(float a) { return lrint(a); }
long int test_lrint(float a) { return lrint(a); }

// CHECK: long long int test_llrint(float a) { return llrint(a); }
long long int test_llrint(float a) { return llrint(a); }

// CHECK: float test_nearbyint(float a) { return nearbyint(a); }
float test_nearbyint(float a) { return nearbyint(a); }

// CHECK: float test_fdim(float a, float b) { return fdim(a, b); }
float test_fdim(float a, float b) { return fdim(a, b); }

// CHECK: float test_fma(float a, float b, float c) { return fma(a, b, c); }
float test_fma(float a, float b, float c) { return fma(a, b, c); }

// CHECK: float test_fmax(float a, float b) { return fmax(a, b); }
float test_fmax(float a, float b) { return fmax(a, b); }

// CHECK: float test_fmin(float a, float b) { return fmin(a, b); }
float test_fmin(float a, float b) { return fmin(a, b); }

// CHECK: float test_exp10(float a) { return exp10(a); }
float test_exp10(float a) { return exp10(a); }

// CHECK: float test_rsqrt(float a) { return rsqrt(a); }
float test_rsqrt(float a) { return rsqrt(a); }

// CHECK: float test_rcbrt(float a) { return rcbrt(a); }
float test_rcbrt(float a) { return rcbrt(a); }

// CHECK: float test_sinpi(float a) { return sinpi(a); }
float test_sinpi(float a) { return sinpi(a); }

// CHECK: float test_cospi(float a) { return cospi(a); }
float test_cospi(float a) { return cospi(a); }

// CHECK: void test_sincospi(float a, float *sptr, float *cptr) {
// CHECK:   return sincospi(a, sptr, cptr);
// CHECK: }
void test_sincospi(float a, float *sptr, float *cptr) {
  return sincospi(a, sptr, cptr);
}

// CHECK: void test_sincos(float a, float *sptr, float *cptr) {
// CHECK:   return sincos(a, sptr, cptr);
// CHECK: }
void test_sincos(float a, float *sptr, float *cptr) {
  return sincos(a, sptr, cptr);
}

// CHECK: float test_j0(float a) { return j0(a); }
float test_j0(float a) { return j0(a); }

// CHECK: float test_j1(float a) { return j1(a); }
float test_j1(float a) { return j1(a); }

// CHECK: float test_jn(int n, float a) { return jn(n, a); }
float test_jn(int n, float a) { return jn(n, a); }

// CHECK: float test_y0(float a) { return y0(a); }
float test_y0(float a) { return y0(a); }

// CHECK: float test_y1(float a) { return y1(a); }
float test_y1(float a) { return y1(a); }

// CHECK: float test_yn(int n, float a) { return yn(n, a); }
float test_yn(int n, float a) { return yn(n, a); }

// CHECK: float test_cyl_bessel_i0(float a) { return cyl_bessel_i0(a); }
float test_cyl_bessel_i0(float a) { return cyl_bessel_i0(a); }

// CHECK: float test_cyl_bessel_i1(float a) { return cyl_bessel_i1(a); }
float test_cyl_bessel_i1(float a) { return cyl_bessel_i1(a); }

// CHECK: float test_erfinv(float a) { return erfinv(a); }
float test_erfinv(float a) { return erfinv(a); }

// CHECK: float test_erfcinv(float a) { return erfcinv(a); }
float test_erfcinv(float a) { return erfcinv(a); }

// CHECK: float test_normcdfinv(float a) { return normcdfinv(a); }
float test_normcdfinv(float a) { return normcdfinv(a); }

// CHECK: float test_normcdf(float a) { return normcdf(a); }
float test_normcdf(float a) { return normcdf(a); }

// CHECK: float test_erfcx(float a) { return erfcx(a); }
float test_erfcx(float a) { return erfcx(a); }

// CHECK: double test_copysign(double a, float b) { return copysign(a, b); }
double test_copysign(double a, float b) { return copysign(a, b); }

// CHECK: double test_copysign(float a, double b) { return copysign(a, b); }
double test_copysign(float a, double b) { return copysign(a, b); }

// CHECK: unsigned int test_min(unsigned int a, unsigned int b) { return std::min(a, b); }
unsigned int test_min(unsigned int a, unsigned int b) { return min(a, b); }

// CHECK: unsigned int test_min(int a, unsigned int b) { return std::min<unsigned int>(a, b); }
unsigned int test_min(int a, unsigned int b) { return min(a, b); }

// CHECK: unsigned int test_min(unsigned int a, int b) { return std::min<unsigned int>(a, b); }
unsigned int test_min(unsigned int a, int b) { return min(a, b); }

// CHECK: long int test_min(long int a, long int b) { return std::min(a, b); }
long int test_min(long int a, long int b) { return min(a, b); }

// CHECK: unsigned long int test_min(unsigned long int a, unsigned long int b) {
// CHECK:   return std::min(a, b);
// CHECK: }
unsigned long int test_min(unsigned long int a, unsigned long int b) {
  return min(a, b);
}

// CHECK: unsigned long int test_min(long int a, unsigned long int b) {
// CHECK:   return std::min<unsigned long>(a, b);
// CHECK: }
unsigned long int test_min(long int a, unsigned long int b) {
  return min(a, b);
}

// CHECK: unsigned long int test_min(unsigned long int a, long int b) {
// CHECK:   return std::min<unsigned long>(a, b);
// CHECK: }
unsigned long int test_min(unsigned long int a, long int b) {
  return min(a, b);
}

// CHECK: long long int test_min(long long int a, long long int b) { return std::min(a, b); }
long long int test_min(long long int a, long long int b) { return min(a, b); }

// CHECK: unsigned long long int test_min(unsigned long long int a,
// CHECK:                                 unsigned long long int b) {
// CHECK:   return std::min(a, b);
// CHECK: }
unsigned long long int test_min(unsigned long long int a,
                                unsigned long long int b) {
  return min(a, b);
}

// CHECK: unsigned long long int test_min(long long int a, unsigned long long int b) {
// CHECK:   return std::min<unsigned long long>(a, b);
// CHECK: }
unsigned long long int test_min(long long int a, unsigned long long int b) {
  return min(a, b);
}

// CHECK: unsigned long long int test_min(unsigned long long int a, long long int b) {
// CHECK:   return std::min<unsigned long long>(a, b);
// CHECK: }
unsigned long long int test_min(unsigned long long int a, long long int b) {
  return min(a, b);
}

// CHECK: float test_min(float a, float b) { return fminf(a, b); }
float test_min(float a, float b) { return min(a, b); }

// CHECK: double test_min(double a, double b) { return fmin(a, b); }
double test_min(double a, double b) { return min(a, b); }

// CHECK: double test_min(float a, double b) { return fmin(a, b); }
double test_min(float a, double b) { return min(a, b); }

// CHECK: double test_min(double a, float b) { return fmin(a, b); }
double test_min(double a, float b) { return min(a, b); }

// CHECK: unsigned int test_max(unsigned int a, unsigned int b) { return std::max(a, b); }
unsigned int test_max(unsigned int a, unsigned int b) { return max(a, b); }

// CHECK: unsigned int test_max(int a, unsigned int b) { return std::max<unsigned int>(a, b); }
unsigned int test_max(int a, unsigned int b) { return max(a, b); }

// CHECK: unsigned int test_max(unsigned int a, int b) { return std::max<unsigned int>(a, b); }
unsigned int test_max(unsigned int a, int b) { return max(a, b); }

// CHECK: long int test_max(long int a, long int b) { return std::max(a, b); }
long int test_max(long int a, long int b) { return max(a, b); }

// CHECK: unsigned long int test_max(unsigned long int a, unsigned long int b) {
// CHECK:   return std::max(a, b);
// CHECK: }
unsigned long int test_max(unsigned long int a, unsigned long int b) {
  return max(a, b);
}

// CHECK: unsigned long int test_max(long int a, unsigned long int b) {
// CHECK:   return std::max<unsigned long>(a, b);
// CHECK: }
unsigned long int test_max(long int a, unsigned long int b) {
  return max(a, b);
}

// CHECK: unsigned long int test_max(unsigned long int a, long int b) {
// CHECK:   return std::max<unsigned long>(a, b);
// CHECK: }
unsigned long int test_max(unsigned long int a, long int b) {
  return max(a, b);
}

// CHECK: long long int test_max(long long int a, long long int b) { return std::max(a, b); }
long long int test_max(long long int a, long long int b) { return max(a, b); }

// CHECK: unsigned long long int test_max(unsigned long long int a,
// CHECK:                                 unsigned long long int b) {
// CHECK:   return std::max(a, b);
// CHECK: }
unsigned long long int test_max(unsigned long long int a,
                                unsigned long long int b) {
  return max(a, b);
}

// CHECK: unsigned long long int test_max(long long int a, unsigned long long int b) {
// CHECK:   return std::max<unsigned long long>(a, b);
// CHECK: }
unsigned long long int test_max(long long int a, unsigned long long int b) {
  return max(a, b);
}

// CHECK: unsigned long long int test_max(unsigned long long int a, long long int b) {
// CHECK:   return std::max<unsigned long long>(a, b);
// CHECK: }
unsigned long long int test_max(unsigned long long int a, long long int b) {
  return max(a, b);
}

// CHECK: float test_max(float a, float b) { return fmaxf(a, b); }
float test_max(float a, float b) { return max(a, b); }

// CHECK: double test_max(double a, double b) { return fmax(a, b); }
double test_max(double a, double b) { return max(a, b); }

// CHECK: double test_max(float a, double b) { return fmax(a, b); }
double test_max(float a, double b) { return max(a, b); }

// CHECK: double test_max(double a, float b) { return fmax(a, b); }
double test_max(double a, float b) { return max(a, b); }

// max/min() without argments bellow are differnt with max(a,b)/min(a,b).
void foo_1() {
  // CHECK: unsigned long long max = (unsigned long long) std::numeric_limits<int>::max();
  // CHECK: unsigned long long min = (unsigned long long) std::numeric_limits<int>::min();
  unsigned long long max = (unsigned long long) std::numeric_limits<int>::max();
  unsigned long long min = (unsigned long long) std::numeric_limits<int>::min();
}

template<typename T>
void foo_2(T t)
{
  // CHECK: unsigned long long max = (unsigned long long) std::numeric_limits<int>::max();
  // CHECK: unsigned long long min = (unsigned long long) std::numeric_limits<int>::min();
  unsigned long long max = (unsigned long long) std::numeric_limits<int>::max();
  unsigned long long min = (unsigned long long) std::numeric_limits<int>::min();
}

// CHECK: #define MAX_INT (unsigned long long) std::numeric_limits<int>::max()
// CHECK: #define MIN_INT (unsigned long long) std::numeric_limits<int>::min()
#define MAX_INT (unsigned long long) std::numeric_limits<int>::max()
#define MIN_INT (unsigned long long) std::numeric_limits<int>::min()

#define foo_inner(x) (x)
void foo_3(){
    // CHECK: unsigned long long max = foo_inner((unsigned long long) std::numeric_limits<int>::max());
    // CHECK: unsigned long long min = foo_inner((unsigned long long) std::numeric_limits<int>::min());
  unsigned long long max = foo_inner((unsigned long long) std::numeric_limits<int>::max());
  unsigned long long min = foo_inner((unsigned long long) std::numeric_limits<int>::min());
}

