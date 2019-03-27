// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cuda-math-intrinsics.sycl.cpp --match-full-lines %s

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <stdio.h>
using namespace std;

__global__ void kernelFuncDouble(double *deviceArrayDouble) {
  double &d0 = *deviceArrayDouble, &d1 = *(deviceArrayDouble + 1), &d2 = *(deviceArrayDouble + 2);
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

  // CHECK: f0 = cl::sycl::cos(f0);
  f0 = __cosf(f0);
  // CHECK: f0 = cl::sycl::exp10(f0);
  f0 = __exp10f(f0);
  // CHECK: f0 = cl::sycl::exp(f0);
  f0 = __expf(f0);
  // CHECK: f2 = cl::sycl::divide(f0, f1);
  f2 = __fdividef(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1013:{{[0-9]+}}: The rounding mode of {{[a-zA-z:\+\-\*\/]+}} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = cl::sycl::rsqrt(f2);
  f2 = __frsqrt_rn(f2);
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

int main() {
  testDouble();
  testFloat();
}
