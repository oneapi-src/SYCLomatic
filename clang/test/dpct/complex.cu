// RUN: dpct --format-range=none -out-root %T/complex %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/complex/complex.dp.cpp

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>

// CHECK: #define COMPLEX_D_MAKE(r,i) sycl::double2(r, i)
// CHECK: #define COMPLEX_D_REAL(a) (a).x()
// CHECK: #define COMPLEX_D_IMAG(a) (a).y()
// CHECK: #define COMPLEX_D_FREAL(a) a.x()
// CHECK: #define COMPLEX_D_FIMAG(a) a.y()
// CHECK: #define COMPLEX_D_ADD(a, b) a + b
// CHECK: #define COMPLEX_D_SUB(a, b) a - b
// CHECK: #define COMPLEX_D_MUL(a, b) dpct::cmul<double>(a, b)
// CHECK: #define COMPLEX_D_DIV(a, b) dpct::cdiv<double>(a, b)
// CHECK: #define COMPLEX_D_ABS(a) dpct::cabs<double>(a)
// CHECK: #define COMPLEX_D_ABS1(a) (sycl::fabs((a).x()) + sycl::fabs((a).y()))
// CHECK: #define COMPLEX_D_CONJ(a) dpct::conj<double>(a)
#define COMPLEX_D_MAKE(r,i) make_cuDoubleComplex(r, i)
#define COMPLEX_D_REAL(a) (a).x
#define COMPLEX_D_IMAG(a) (a).y
#define COMPLEX_D_FREAL(a) cuCreal(a)
#define COMPLEX_D_FIMAG(a) cuCimag(a)
#define COMPLEX_D_ADD(a, b) cuCadd(a, b)
#define COMPLEX_D_SUB(a, b) cuCsub(a, b)
#define COMPLEX_D_MUL(a, b) cuCmul(a, b)
#define COMPLEX_D_DIV(a, b) cuCdiv(a, b)
#define COMPLEX_D_ABS(a) cuCabs(a)
#define COMPLEX_D_ABS1(a) (fabs((a).x) + fabs((a).y))
#define COMPLEX_D_CONJ(a) cuConj(a)

// CHECK: #define COMPLEX_F_MAKE(r,i) sycl::float2(r, i)
// CHECK: #define COMPLEX_F_REAL(a) (a).x()
// CHECK: #define COMPLEX_F_IMAG(a) (a).y()
// CHECK: #define COMPLEX_F_FREAL(a) a.x()
// CHECK: #define COMPLEX_F_FIMAG(a) a.y()
// CHECK: #define COMPLEX_F_ADD(a, b) a + b
// CHECK: #define COMPLEX_F_SUB(a, b) a - b
// CHECK: #define COMPLEX_F_MUL(a, b) dpct::cmul<float>(a, b)
// CHECK: #define COMPLEX_F_DIV(a, b) dpct::cdiv<float>(a, b)
// CHECK: #define COMPLEX_F_ABS(a) dpct::cabs<float>(a)
// CHECK: #define COMPLEX_F_ABS1(a) (sycl::fabs((a).x()) + sycl::fabs((a).y()))
// CHECK: #define COMPLEX_F_CONJ(a) dpct::conj<float>(a)
#define COMPLEX_F_MAKE(r,i) make_cuFloatComplex(r, i)
#define COMPLEX_F_REAL(a) (a).x
#define COMPLEX_F_IMAG(a) (a).y
#define COMPLEX_F_FREAL(a) cuCrealf(a)
#define COMPLEX_F_FIMAG(a) cuCimagf(a)
#define COMPLEX_F_ADD(a, b) cuCaddf(a, b)
#define COMPLEX_F_SUB(a, b) cuCsubf(a, b)
#define COMPLEX_F_MUL(a, b) cuCmulf(a, b)
#define COMPLEX_F_DIV(a, b) cuCdivf(a, b)
#define COMPLEX_F_ABS(a) cuCabsf(a)
#define COMPLEX_F_ABS1(a) (fabsf((a).x) + fabsf((a).y))
#define COMPLEX_F_CONJ(a) cuConjf(a)

template <typename T>
__host__ __device__ bool check(T x, float e[], int& index) {
    if((std::abs(x.x - e[index++]) < 0.001) && (std::abs(x.y - e[index++]) < 0.001)) {
        return true;
    }
    return false;
}

template <>
__host__ __device__ bool check<float>(float x, float e[], int& index) {
  if(std::abs(x - e[index++]) < 0.001) {
      return true;
  }
  return false;
}

template <>
__host__ __device__ bool check<double>(double x, float e[], int& index) {
  if(std::abs(x - e[index++]) < 0.001) {
      return true;
  }
  return false;
}

__global__ void kernel(int *result) {
    // CHECK: sycl::float2 f1, f2;
    // CHECK: sycl::double2 d1, d2;
    cuFloatComplex f1, f2;
    cuDoubleComplex d1, d2;
    // CHECK: f1 = COMPLEX_F_MAKE(1.8, -2.7);
    // CHECK: f2 = COMPLEX_F_MAKE(-3.6, 4.5);
    // CHECK: d1 = COMPLEX_D_MAKE(5.4, -6.3);
    // CHECK: d2 = COMPLEX_D_MAKE(-7.2, 8.1);
    f1 = COMPLEX_F_MAKE(1.8, -2.7);
    f2 = COMPLEX_F_MAKE(-3.6, 4.5);
    d1 = COMPLEX_D_MAKE(5.4, -6.3);
    d2 = COMPLEX_D_MAKE(-7.2, 8.1);
   
    int index = 0;
    bool r = true;
    float expect[32] = {5.400000,  -6.300000,
        5.400000,  -6.300000, -1.800000, 1.800000, 12.600000, -14.400000, 12.150000,
        89.100000, -0.765517, 0.013793,  8.297590, 11.700000, 5.400000,   6.300000,
        1.800000,  -2.700000, -1.800000, 1.800000, 5.400000,  -7.200000,  5.670001,
        17.820000, -0.560976, 0.048780,  3.244996, 4.500000,  1.800000,   2.700000,
        1.800000,   -2.700000};
    // CHECK: auto a1 = COMPLEX_D_FREAL(d1);
    auto a1 = COMPLEX_D_FREAL(d1);
    r = r && check(a1, expect, index);
    // CHECK: auto a2 = COMPLEX_D_FIMAG(d1);
    auto a2 = COMPLEX_D_FIMAG(d1);
    r = r && check(a2, expect, index);
    // CHECK: auto a3 = COMPLEX_D_REAL(d1);
    auto a3 = COMPLEX_D_REAL(d1);
    r = r && check(a3, expect, index);
    // CHECK: auto a4 = COMPLEX_D_IMAG(d1);
    auto a4 = COMPLEX_D_IMAG(d1);
    r = r && check(a4, expect, index);
    // CHECK: auto a5 = COMPLEX_D_ADD(d1, d2);
    auto a5 = COMPLEX_D_ADD(d1, d2);
    r = r && check(a5, expect, index);
    // CHECK: auto a6 = COMPLEX_D_SUB(d1, d2);
    auto a6 = COMPLEX_D_SUB(d1, d2);
    r = r && check(a6, expect, index);
    // CHECK: auto a7 = COMPLEX_D_MUL(d1, d2);
    auto a7 = COMPLEX_D_MUL(d1, d2);
    r = r && check(a7, expect, index);
    // CHECK: auto a8 = COMPLEX_D_DIV(d1, d2);
    auto a8 = COMPLEX_D_DIV(d1, d2);
    r = r && check(a8, expect, index);
    // CHECK: auto a9 = COMPLEX_D_ABS(d1);
    auto a9 = COMPLEX_D_ABS(d1);
    r = r && check(a9, expect, index);
    // CHECK: auto a10 = COMPLEX_D_ABS1(d1);
    auto a10 = COMPLEX_D_ABS1(d1);
    r = r && check(a10, expect, index);
    // CHECK: auto a11 = COMPLEX_D_CONJ(d1);
    auto a11 = COMPLEX_D_CONJ(d1);
    r = r && check(a11, expect, index);

    // CHECK: auto a13 = COMPLEX_F_REAL(f1);
    auto a13 = COMPLEX_F_REAL(f1);
    r = r && check(a13, expect, index);
    // CHECK: auto a14 = COMPLEX_F_IMAG(f1);
    auto a14 = COMPLEX_F_IMAG(f1);
    r = r && check(a14, expect, index);
    // CHECK: auto a15 = COMPLEX_F_ADD(f1, f2);
    auto a15 = COMPLEX_F_ADD(f1, f2);
    r = r && check(a15, expect, index);
    // CHECK: auto a16 = COMPLEX_F_SUB(f1, f2);
    auto a16 = COMPLEX_F_SUB(f1, f2);
    r = r && check(a16, expect, index);
    // CHECK: auto a17 = COMPLEX_F_MUL(f1, f2);
    auto a17 = COMPLEX_F_MUL(f1, f2);
    r = r && check(a17, expect, index);
    // CHECK: auto a18 = COMPLEX_F_DIV(f1, f2);
    auto a18 = COMPLEX_F_DIV(f1, f2);
    r = r && check(a18, expect, index);
    // CHECK: auto a19 = COMPLEX_F_ABS(f1);
    auto a19 = COMPLEX_F_ABS(f1);
    r = r && check(a19, expect, index);
    // CHECK: auto a20 = COMPLEX_F_ABS1(f1);
    auto a20 = COMPLEX_F_ABS1(f1);
    r = r && check(a20, expect, index);
    // CHECK: auto a21 = COMPLEX_F_CONJ(f1);
    auto a21 = COMPLEX_F_CONJ(f1);
    r = r && check(a21, expect, index);
    // CHECK: auto a22 = COMPLEX_F_FREAL(f1);
    auto a22 = COMPLEX_F_FREAL(f1);
    r = r && check(a22, expect, index);
    // CHECK: auto a23 = COMPLEX_F_FIMAG(f1);
    auto a23 = COMPLEX_F_FIMAG(f1);
    r = r && check(a23, expect, index);
    *result = r;
}

int main() {
    // CHECK: sycl::float2 f1, f2;
    // CHECK: sycl::double2 d1, d2;
    cuFloatComplex f1, f2;
    cuDoubleComplex d1, d2;
    // CHECK: f1 = COMPLEX_F_MAKE(1.8, -2.7);
    // CHECK: f2 = COMPLEX_F_MAKE(-3.6, 4.5);
    // CHECK: d1 = COMPLEX_D_MAKE(5.4, -6.3);
    // CHECK: d2 = COMPLEX_D_MAKE(-7.2, 8.1);
    f1 = COMPLEX_F_MAKE(1.8, -2.7);
    f2 = COMPLEX_F_MAKE(-3.6, 4.5);
    d1 = COMPLEX_D_MAKE(5.4, -6.3);
    d2 = COMPLEX_D_MAKE(-7.2, 8.1);
    int index = 0;
    bool r = true;
    float expect[32] = {5.400000,  -6.300000,
        5.400000,  -6.300000, -1.800000, 1.800000, 12.600000, -14.400000, 12.150000,
        89.100000, -0.765517, 0.013793,  8.297590, 11.700000, 5.400000,   6.300000,
        1.800000,  -2.700000, -1.800000, 1.800000, 5.400000,  -7.200000,  5.670001,
        17.820000, -0.560976, 0.048780,  3.244996, 4.500000,  1.800000,   2.700000,
        1.800000,   -2.700000};
    // CHECK: auto a1 = COMPLEX_D_FREAL(d1);
    auto a1 = COMPLEX_D_FREAL(d1);
    r = r && check(a1, expect, index);
    // CHECK: auto a2 = COMPLEX_D_FIMAG(d1);
    auto a2 = COMPLEX_D_FIMAG(d1);
    r = r && check(a2, expect, index);
    // CHECK: auto a3 = COMPLEX_D_REAL(d1);
    auto a3 = COMPLEX_D_REAL(d1);
    r = r && check(a3, expect, index);
    // CHECK: auto a4 = COMPLEX_D_IMAG(d1);
    auto a4 = COMPLEX_D_IMAG(d1);
    r = r && check(a4, expect, index);
    // CHECK: auto a5 = COMPLEX_D_ADD(d1, d2);
    auto a5 = COMPLEX_D_ADD(d1, d2);
    r = r && check(a5, expect, index);
    // CHECK: auto a6 = COMPLEX_D_SUB(d1, d2);
    auto a6 = COMPLEX_D_SUB(d1, d2);
    r = r && check(a6, expect, index);
    // CHECK: auto a7 = COMPLEX_D_MUL(d1, d2);
    auto a7 = COMPLEX_D_MUL(d1, d2);
    r = r && check(a7, expect, index);
    // CHECK: auto a8 = COMPLEX_D_DIV(d1, d2);
    auto a8 = COMPLEX_D_DIV(d1, d2);
    r = r && check(a8, expect, index);
    // CHECK: auto a9 = COMPLEX_D_ABS(d1);
    auto a9 = COMPLEX_D_ABS(d1);
    r = r && check(a9, expect, index);
    // CHECK: auto a10 = COMPLEX_D_ABS1(d1);
    auto a10 = COMPLEX_D_ABS1(d1);
    r = r && check(a10, expect, index);
    // CHECK: auto a11 = COMPLEX_D_CONJ(d1);
    auto a11 = COMPLEX_D_CONJ(d1);
    r = r && check(a11, expect, index);

    // CHECK: auto a13 = COMPLEX_F_REAL(f1);
    auto a13 = COMPLEX_F_REAL(f1);
    r = r && check(a13, expect, index);
    // CHECK: auto a14 = COMPLEX_F_IMAG(f1);
    auto a14 = COMPLEX_F_IMAG(f1);
    r = r && check(a14, expect, index);
    // CHECK: auto a15 = COMPLEX_F_ADD(f1, f2);
    auto a15 = COMPLEX_F_ADD(f1, f2);
    r = r && check(a15, expect, index);
    // CHECK: auto a16 = COMPLEX_F_SUB(f1, f2);
    auto a16 = COMPLEX_F_SUB(f1, f2);
    r = r && check(a16, expect, index);
    // CHECK: auto a17 = COMPLEX_F_MUL(f1, f2);
    auto a17 = COMPLEX_F_MUL(f1, f2);
    r = r && check(a17, expect, index);
    // CHECK: auto a18 = COMPLEX_F_DIV(f1, f2);
    auto a18 = COMPLEX_F_DIV(f1, f2);
    r = r && check(a18, expect, index);
    // CHECK: auto a19 = COMPLEX_F_ABS(f1);
    auto a19 = COMPLEX_F_ABS(f1);
    r = r && check(a19, expect, index);
    // CHECK: auto a20 = COMPLEX_F_ABS1(f1);
    auto a20 = COMPLEX_F_ABS1(f1);
    r = r && check(a20, expect, index);
    // CHECK: auto a21 = COMPLEX_F_CONJ(f1);
    auto a21 = COMPLEX_F_CONJ(f1);
    r = r && check(a21, expect, index);
    // CHECK: auto a22 = COMPLEX_F_FREAL(f1);
    auto a22 = COMPLEX_F_FREAL(f1);
    r = r && check(a22, expect, index);
    // CHECK: auto a23 = COMPLEX_F_FIMAG(f1);
    auto a23 = COMPLEX_F_FIMAG(f1);
    r = r && check(a23, expect, index);

    int *result = nullptr;
    cudaMallocManaged(&result, sizeof(int));
    *result = 0;

    kernel<<<1,1>>>(result);
    cudaDeviceSynchronize();

    if(*result && r) {
      std::cout << "pass" << std::endl;
    } else {
      std::cout << "fail" << std::endl;
      exit(-1);
    }
    return 0;
}
  
void testRemoveVolatile() {
  struct Complex : cuFloatComplex {

    inline float real() const volatile {
      // CHECK: /*
      // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
      // CHECK-NEXT: */
      // CHECK-NEXT: return const_cast<struct Complex *>(this)->x();
      return x;
    }

    inline float imag() const volatile {
      // CHECK: /*
      // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
      // CHECK-NEXT: */
      // CHECK-NEXT: return const_cast<struct Complex *>(this)->y();
      return y;
    }

    inline void testExplicit() const volatile {
      // CHECK: /*
      // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
      // CHECK-NEXT: */
      // CHECK-NEXT: const_cast<struct Complex *>(this)->y();
      this->y;

      auto self = this;
      // CHECK: /*
      // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
      // CHECK-NEXT: */
      // CHECK-NEXT: const_cast<struct Complex *>(self)->y();
      self->y;
    }

  };
}
