// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/math_functions_test.sycl.cpp --match-full-lines %s

#include <cuda.h>
#include <cmath>

// CHECK: int test_signbit(float x) try { return signbit(x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_signbit(float x) { return signbit(x); }

// CHECK: int test_signbit(double x) try { return signbit(x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_signbit(double x) { return signbit(x); }

// CHECK: int test_signbit(long double x) try { return signbit((double)x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_signbit(long double x) { return signbit((double)x); }

// CHECK: int test_isfinite(float x) try { return isfinite(x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isfinite(float x) { return isfinite(x); }

// CHECK: int test_isfinite(double x) try { return isfinite(x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isfinite(double x) { return isfinite(x); }

// CHECK: int test_isfinite(long double x) try { return isfinite((double)x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isfinite(long double x) { return isfinite((double)x); }

// CHECK: int test_isnan(float x) try { return isnan(x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isnan(float x) { return isnan(x); }

// CHECK: int test_isnan(double x) try { return isnan(x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isnan(double x) { return isnan(x); }

// CHECK: int test_isnan(long double x) try { return isnan((double)x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isnan(long double x) { return isnan((double)x); }

// CHECK: int test_isinf(float x) try { return isinf(x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isinf(float x) { return isinf(x); }

// CHECK: int test_isinf(double x) try { return isinf(x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isinf(double x) { return isinf(x); }

// CHECK: int test_isinf(long double x) try { return isinf((double)x); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_isinf(long double x) { return isinf((double)x); }

// CHECK: long long int test_abs(long long int a) try { return abs(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long long int test_abs(long long int a) { return abs(a); }

// CHECK: long int test_abs(long int in) try { return abs(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long int test_abs(long int in) { return abs(in); }

// The check for this will be enabled, when the patch for CTST-689 is merged.
float test_abs(float in) { return abs(in); }

// The check for this will be enabled, when the patch for CTST-689 is merged.
double test_abs(double in) { return abs(in); }

// CHECK: float test_fabs(float in) try { return fabs(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_fabs(float in) { return fabs(in); }

// CHECK: float test_ceil(float in) try { return ceil(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_ceil(float in) { return ceil(in); }

// CHECK: float test_floor(float in) try { return floor(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_floor(float in) { return floor(in); }

// CHECK: float test_sqrt(float in) try { return sqrt(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_sqrt(float in) { return sqrt(in); }

// CHECK: float test_pow(float a, float b) try { return pow(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_pow(float a, float b) { return pow(a, b); }

// CHECK: float test_pow(float a, int b) try { return pow(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_pow(float a, int b) { return pow(a, b); }

// CHECK: double test_pow(double a, int b) try { return pow(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_pow(double a, int b) { return pow(a, b); }

// CHECK: float test_powif(float a, int b) try { return powif(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_powif(float a, int b) { return powif(a, b); }

// CHECK: double test_powi(double a, int b) try { return powi(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_powi(double a, int b) { return powi(a, b); }

// CHECK: float test_log(float in) try { return log(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_log(float in) { return log(in); }

// CHECK: float test_log10(float in) try { return log10(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_log10(float in) { return log10(in); }

// CHECK: float test_fmod(float a, float b) try { return fmod(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_fmod(float a, float b) { return fmod(a, b); }

// CHECK: float test_modf(float a, float *b) try { return modf(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_modf(float a, float *b) { return modf(a, b); }

// CHECK: float test_exp(float in) try { return exp(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_exp(float in) { return exp(in); }

// CHECK: float test_frexp(float a, int *b) try { return frexp(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_frexp(float a, int *b) { return frexp(a, b); }

// CHECK: float test_ldexp(float a, int b) try { return ldexp(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_ldexp(float a, int b) { return ldexp(a, b); }

// CHECK: float test_asin(float in) try { return asin(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_asin(float in) { return asin(in); }

// CHECK: float test_sin(float in) try { return sin(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_sin(float in) { return sin(in); }

// CHECK: float test_sinh(float in) try { return sinh(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_sinh(float in) { return sinh(in); }

// CHECK: float test_acos(float in) try { return acos(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_acos(float in) { return acos(in); }

// CHECK: float test_cos(float in) try { return cos(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_cos(float in) { return cos(in); }

// CHECK: float test_cosh(float in) try { return cosh(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_cosh(float in) { return cosh(in); }

// CHECK: float test_atan(float in) try { return atan(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_atan(float in) { return atan(in); }

// CHECK: float test_atan2(float a, float b) try { return atan2(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_atan2(float a, float b) { return atan2(a, b); }

// CHECK: float test_tan(float in) try { return tan(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_tan(float in) { return tan(in); }

// CHECK: float test_tanh(float in) try { return tanh(in); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_tanh(float in) { return tanh(in); }

// CHECK: float test_logb(float a) try { return logb(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_logb(float a) { return logb(a); }

// CHECK: int test_ilogb(float a) try { return ilogb(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
int test_ilogb(float a) { return ilogb(a); }

// CHECK: float test_scalbn(float a, int b) try { return scalbn(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_scalbn(float a, int b) { return scalbn(a, b); }

// CHECK: float test_scalbln(float a, long int b) try { return scalbln(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_scalbln(float a, long int b) { return scalbln(a, b); }

// CHECK: float test_exp2(float a) try { return exp2(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_exp2(float a) { return exp2(a); }

// CHECK: float test_expm1(float a) try { return expm1(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_expm1(float a) { return expm1(a); }

// CHECK: float test_log2(float a) try { return log2(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_log2(float a) { return log2(a); }

// CHECK: float test_log1p(float a) try { return log1p(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_log1p(float a) { return log1p(a); }

// CHECK: float test_acosh(float a) try { return acosh(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_acosh(float a) { return acosh(a); }

// CHECK: float test_asinh(float a) try { return asinh(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_asinh(float a) { return asinh(a); }

// CHECK: float test_atanh(float a) try { return atanh(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_atanh(float a) { return atanh(a); }

// CHECK: float test_hypot(float a, float b) try { return hypot(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_hypot(float a, float b) { return hypot(a, b); }

// CHECK: float test_norm3d(float a, float b, float c) try { return norm3d(a, b, c); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_norm3d(float a, float b, float c) { return norm3d(a, b, c); }

// CHECK: float test_norm4d(float a, float b, float c, float d) try {
// CHECK:   return norm4d(a, b, c, d);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_norm4d(float a, float b, float c, float d) {
  return norm4d(a, b, c, d);
}

// CHECK: float test_cbrt(float a) try { return cbrt(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_cbrt(float a) { return cbrt(a); }

// CHECK: float test_erf(float a) try { return erf(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_erf(float a) { return erf(a); }

// CHECK: float test_erfc(float a) try { return erfc(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_erfc(float a) { return erfc(a); }

// CHECK: float test_lgamma(float a) try { return lgamma(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_lgamma(float a) { return lgamma(a); }

// CHECK: float test_tgamma(float a) try { return tgamma(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_tgamma(float a) { return tgamma(a); }

// CHECK: float test_copysign(float a, float b) try { return copysign(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_copysign(float a, float b) { return copysign(a, b); }

// CHECK: float test_nextafter(float a, float b) try { return nextafter(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_nextafter(float a, float b) { return nextafter(a, b); }

// CHECK: float test_remainder(float a, float b) try { return remainder(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_remainder(float a, float b) { return remainder(a, b); }

// CHECK: float test_remquo(float a, float b, int *quo) try { return remquo(a, b, quo); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_remquo(float a, float b, int *quo) { return remquo(a, b, quo); }

// CHECK: float test_round(float a) try { return round(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_round(float a) { return round(a); }

// CHECK: long int test_lround(float a) try { return lround(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long int test_lround(float a) { return lround(a); }

// CHECK: long long int test_llround(float a) try { return llround(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long long int test_llround(float a) { return llround(a); }

// CHECK: float test_trunc(float a) try { return trunc(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_trunc(float a) { return trunc(a); }

// CHECK: float test_rint(float a) try { return rint(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_rint(float a) { return rint(a); }

// CHECK: long int test_lrint(float a) try { return lrint(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long int test_lrint(float a) { return lrint(a); }

// CHECK: long long int test_llrint(float a) try { return llrint(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long long int test_llrint(float a) { return llrint(a); }

// CHECK: float test_nearbyint(float a) try { return nearbyint(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_nearbyint(float a) { return nearbyint(a); }

// CHECK: float test_fdim(float a, float b) try { return fdim(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_fdim(float a, float b) { return fdim(a, b); }

// CHECK: float test_fma(float a, float b, float c) try { return fma(a, b, c); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_fma(float a, float b, float c) { return fma(a, b, c); }

// CHECK: float test_fmax(float a, float b) try { return fmax(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_fmax(float a, float b) { return fmax(a, b); }

// CHECK: float test_fmin(float a, float b) try { return fmin(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_fmin(float a, float b) { return fmin(a, b); }

// CHECK: float test_exp10(float a) try { return exp10(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_exp10(float a) { return exp10(a); }

// CHECK: float test_rsqrt(float a) try { return rsqrt(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_rsqrt(float a) { return rsqrt(a); }

// CHECK: float test_rcbrt(float a) try { return rcbrt(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_rcbrt(float a) { return rcbrt(a); }

// CHECK: float test_sinpi(float a) try { return sinpi(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_sinpi(float a) { return sinpi(a); }

// CHECK: float test_cospi(float a) try { return cospi(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_cospi(float a) { return cospi(a); }

// CHECK: void test_sincospi(float a, float *sptr, float *cptr) try {
// CHECK:   return sincospi(a, sptr, cptr);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
void test_sincospi(float a, float *sptr, float *cptr) {
  return sincospi(a, sptr, cptr);
}

// CHECK: void test_sincos(float a, float *sptr, float *cptr) try {
// CHECK:   return sincos(a, sptr, cptr);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
void test_sincos(float a, float *sptr, float *cptr) {
  return sincos(a, sptr, cptr);
}

// CHECK: float test_j0(float a) try { return j0(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_j0(float a) { return j0(a); }

// CHECK: float test_j1(float a) try { return j1(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_j1(float a) { return j1(a); }

// CHECK: float test_jn(int n, float a) try { return jn(n, a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_jn(int n, float a) { return jn(n, a); }

// CHECK: float test_y0(float a) try { return y0(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_y0(float a) { return y0(a); }

// CHECK: float test_y1(float a) try { return y1(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_y1(float a) { return y1(a); }

// CHECK: float test_yn(int n, float a) try { return yn(n, a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_yn(int n, float a) { return yn(n, a); }

// CHECK: float test_cyl_bessel_i0(float a) try { return cyl_bessel_i0(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_cyl_bessel_i0(float a) { return cyl_bessel_i0(a); }

// CHECK: float test_cyl_bessel_i1(float a) try { return cyl_bessel_i1(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_cyl_bessel_i1(float a) { return cyl_bessel_i1(a); }

// CHECK: float test_erfinv(float a) try { return erfinv(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_erfinv(float a) { return erfinv(a); }

// CHECK: float test_erfcinv(float a) try { return erfcinv(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_erfcinv(float a) { return erfcinv(a); }

// CHECK: float test_normcdfinv(float a) try { return normcdfinv(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_normcdfinv(float a) { return normcdfinv(a); }

// CHECK: float test_normcdf(float a) try { return normcdf(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_normcdf(float a) { return normcdf(a); }

// CHECK: float test_erfcx(float a) try { return erfcx(a); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_erfcx(float a) { return erfcx(a); }

// CHECK: double test_copysign(double a, float b) try { return copysign(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_copysign(double a, float b) { return copysign(a, b); }

// CHECK: double test_copysign(float a, double b) try { return copysign(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_copysign(float a, double b) { return copysign(a, b); }

// CHECK: unsigned int test_min(unsigned int a, unsigned int b) try { return min(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned int test_min(unsigned int a, unsigned int b) { return min(a, b); }

// CHECK: unsigned int test_min(int a, unsigned int b) try { return min(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned int test_min(int a, unsigned int b) { return min(a, b); }

// CHECK: unsigned int test_min(unsigned int a, int b) try { return min(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned int test_min(unsigned int a, int b) { return min(a, b); }

// CHECK: long int test_min(long int a, long int b) try { return min(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long int test_min(long int a, long int b) { return min(a, b); }

// CHECK: unsigned long int test_min(unsigned long int a, unsigned long int b) try {
// CHECK:   return min(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long int test_min(unsigned long int a, unsigned long int b) {
  return min(a, b);
}

// CHECK: unsigned long int test_min(long int a, unsigned long int b) try {
// CHECK:   return min(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long int test_min(long int a, unsigned long int b) {
  return min(a, b);
}

// CHECK: unsigned long int test_min(unsigned long int a, long int b) try {
// CHECK:   return min(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long int test_min(unsigned long int a, long int b) {
  return min(a, b);
}

// CHECK: long long int test_min(long long int a, long long int b) try { return min(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long long int test_min(long long int a, long long int b) { return min(a, b); }

// CHECK: unsigned long long int test_min(unsigned long long int a,
// CHECK:                                 unsigned long long int b) try {
// CHECK:   return min(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long long int test_min(unsigned long long int a,
                                unsigned long long int b) {
  return min(a, b);
}

// CHECK: unsigned long long int test_min(long long int a, unsigned long long int b) try {
// CHECK:   return min(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long long int test_min(long long int a, unsigned long long int b) {
  return min(a, b);
}

// CHECK: unsigned long long int test_min(unsigned long long int a, long long int b) try {
// CHECK:   return min(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long long int test_min(unsigned long long int a, long long int b) {
  return min(a, b);
}

// CHECK: float test_min(float a, float b) try { return fminf(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_min(float a, float b) { return min(a, b); }

// CHECK: double test_min(double a, double b) try { return fmin(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_min(double a, double b) { return min(a, b); }

// CHECK: double test_min(float a, double b) try { return fminf(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_min(float a, double b) { return min(a, b); }

// CHECK: double test_min(double a, float b) try { return fmin(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_min(double a, float b) { return min(a, b); }

// CHECK: unsigned int test_max(unsigned int a, unsigned int b) try { return max(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned int test_max(unsigned int a, unsigned int b) { return max(a, b); }

// CHECK: unsigned int test_max(int a, unsigned int b) try { return max(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned int test_max(int a, unsigned int b) { return max(a, b); }

// CHECK: unsigned int test_max(unsigned int a, int b) try { return max(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned int test_max(unsigned int a, int b) { return max(a, b); }

// CHECK: long int test_max(long int a, long int b) try { return max(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
long int test_max(long int a, long int b) { return max(a, b); }

// CHECK: unsigned long int test_max(unsigned long int a, unsigned long int b) try {
// CHECK:   return max(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long int test_max(unsigned long int a, unsigned long int b) {
  return max(a, b);
}

// CHECK: unsigned long int test_max(long int a, unsigned long int b) try {
// CHECK:   return max(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long int test_max(long int a, unsigned long int b) {
  return max(a, b);
}

// CHECK: unsigned long int test_max(unsigned long int a, long int b) try {
// CHECK:   return max(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long int test_max(unsigned long int a, long int b) {
  return max(a, b);
}

// CHECK: long long int test_max(long long int a, long long int b) try { return max(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }

long long int test_max(long long int a, long long int b) { return max(a, b); }

// CHECK: unsigned long long int test_max(unsigned long long int a,
// CHECK:                                 unsigned long long int b) try {
// CHECK:   return max(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long long int test_max(unsigned long long int a,
                                unsigned long long int b) {
  return max(a, b);
}

// CHECK: unsigned long long int test_max(long long int a, unsigned long long int b) try {
// CHECK:   return max(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long long int test_max(long long int a, unsigned long long int b) {
  return max(a, b);
}

// CHECK: unsigned long long int test_max(unsigned long long int a, long long int b) try {
// CHECK:   return max(a, b);
// CHECK: }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
unsigned long long int test_max(unsigned long long int a, long long int b) {
  return max(a, b);
}

// CHECK: float test_max(float a, float b) try { return fmaxf(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
float test_max(float a, float b) { return max(a, b); }

// CHECK: double test_max(double a, double b) try { return fmax(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_max(double a, double b) { return max(a, b); }

// CHECK: double test_max(float a, double b) try { return fmaxf(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_max(float a, double b) { return max(a, b); }

// CHECK: double test_max(double a, float b) try { return fmax(a, b); }
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
double test_max(double a, float b) { return max(a, b); }

