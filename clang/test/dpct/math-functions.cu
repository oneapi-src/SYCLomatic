// RUN: dpct --format-range=none --usm-level=none -out-root %T/math-functions %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/math-functions/math-functions.dp.cpp --match-full-lines %s

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
#define DECLARE2LD DECLARE2(long double, ld)

#define DECLARE(type, prefix) \
  type prefix##_a = 0;
#define DECLAREF DECLARE(float, f)
#define DECLARED DECLARE(double, d)
#define DECLAREU DECLARE(unsigned, u)
#define DECLAREL DECLARE(long, l)
#define DECLAREI DECLARE(int, i)
#define DECLARELD DECLARE(long double, ld)

__device__ float4 fun() {
  float4 a, b, c;
  // CHECK: sycl::fma(a.x(), b.x(), c.x());
  __fmaf_rn(a.x, b.x, c.x);
  // CHECK: return sycl::float4(sycl::fma(a.x(), b.x(), c.x()), sycl::fma(a.y(), b.y(), c.y()), sycl::fma(a.z(), b.z(), c.z()), sycl::fma(a.w(), b.w(), c.w()));
  return make_float4(__fmaf_rd(a.x, b.x, c.x), __fmaf_rz(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w));
}

#ifndef USE_STD
#define USING(FUNC) using ::FUNC;
#else
#define USING(FUNC) using std::FUNC;
#endif
// CHECK: using sycl::abs;
USING(abs)

__global__ void kernel() {
  // CHECK: using sycl::abs;
  USING(abs)
}

void host() {
#define USE_STD
// CHECK: using sycl::abs;
  USING(abs)
#undef USE_STD
}

#ifndef USE_STD
#define USING_1(FUNC) using ::FUNC; int a = 1;
#else
#define USING_1(FUNC) using std::FUNC; int a = 1;
#endif
// CHECK: USING_1(abs)
USING_1(abs)

__global__ void kernel_1() {
  // CHECK: USING_1(abs)
  USING_1(abs)
}

void host_1() {
#define USE_STD
// CHECK: USING_1(abs)
  USING_1(abs)
#undef USE_STD
}


void foo() {
  // CHECK:   dpct::get_default_queue().parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, ceil(2.3)), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {
  // CHECK-NEXT:           kernel();
  // CHECK-NEXT:         });
  kernel<<< ceil(2.3), 1 >>>();
}

int main() {
  // max
  {
    DECLARE2F
    DECLARE2D
    DECLARE2U
    DECLARE2L
    DECLARE2I
    DECLARE2LD

    // CHECK: f_b = fmaxf(f_a, f_b);
    f_b = max(f_a, f_b);

    // CHECK: d_b = fmax(d_a, d_b);
    d_b = max(d_a, d_b);

    // CHECK: u_b = std::max(u_a, u_b);
    u_b = max(u_a, u_b);

    // CHECK: i_b = std::max(i_a, i_b);
    i_b = max(i_a, i_b);

    // CHECK: ld_b = fmaxl(ld_a, ld_b);
    ld_b = max(ld_a, ld_b);

    // TODO: Check more primitive type and vector types
  }

  // min
  {
    DECLARE2F
    DECLARE2D
    DECLARE2U
    DECLARE2L
    DECLARE2I
    DECLARE2LD

    // CHECK: f_b = fminf(f_a, f_b);
    f_b = min(f_a, f_b);

    // CHECK: d_b = fmin(d_a, d_b);
    d_b = min(d_a, d_b);

    // CHECK: u_b = std::min(u_a, u_b);
    u_b = min(u_a, u_b);

    // CHECK: i_b = std::min(i_a, i_b);
    i_b = min(i_a, i_b);

    // CHECK: ld_b = fminl(ld_a, ld_b);
    ld_b = min(ld_a, ld_b);

    // TODO: Check more primitive type and vector types
  }

  // abs
  {
    DECLAREF
    DECLARED
    DECLAREU
    DECLAREL
    DECLAREI
    DECLARELD

    // CHECK: f_a = abs(f_a);
    f_a = abs(f_a);

    // CHECK: d_a = abs(d_a);
    d_a = abs(d_a);

    // CHECK: i_a = abs(i_a);
    i_a = abs(i_a);

    // CHECK: ld_a = abs(ld_a);
    ld_a = abs(ld_a);

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

    // CHECK: f_a = exp(f_a);
    f_a = exp(f_a);
    // CHECK: d_a = exp(d_a);
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

    // CHECK: f_a = fabs(f_a);
    f_a = fabs(f_a);
    // CHECK: d_a = fabs(d_a);
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

    // CHECK: f_a = sqrt(f_a);
    f_a = sqrt(f_a);
    // CHECK: d_a = sqrt(d_a);
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

template <typename T> class AccPtr {
  size_t size;

public:
  size_t getSize() { return size; }
};


template <typename T>
__device__ static void multiply(int block_size, AccPtr<T> &ptr, T value) {
  // CHECK:int BSZ = ((int)sycl::ceil((float)ptr.getSize() / (float)block_size));
  int BSZ = ((int)ceilf((float)ptr.getSize() / (float)block_size));
}

__device__ void sincos_1(double x, double* sptr, double* cptr) {
  // CHECK:  return [&](){ *(sptr) = sycl::sincos(x, sycl::make_ptr<double, sycl::access::address_space::global_space>(cptr)); }();
  return ::sincos(x, sptr, cptr);
}

__device__ void sincospi_1(double x, double* sptr, double* cptr) {
  // CHECK:  return [&](){ *(sptr) = sycl::sincos(x * DPCT_PI, sycl::make_ptr<double, sycl::access::address_space::global_space>(cptr)); }();
  return ::sincospi (x, sptr, cptr);
}

template <typename T>
__device__  T foo2(T a) {
  // CHECK: return sycl::rsqrt((double)a);
  return ::rsqrt(a);
}

// CHECK: template <class T>
// CHECK-NEXT: T foo3(T a) {
// CHECK-NEXT:   return sycl::cos((float)a);
// CHECK-NEXT: }
template <class T> 
__device__ T foo3(T a) {
  return __cosf(a);
}
__global__ void foo4() {
  double d1, d2;
  d2 = foo3(d1);
}
