// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/cuda-math-extension %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda-math-extension/cuda-math-extension.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncDouble(double *deviceArrayDouble) {
  double &d0 = *deviceArrayDouble, &d1 = *(deviceArrayDouble + 1), &d2 = *(deviceArrayDouble + 2);
  int i;

  // Double Precision Mathematical Functions

  // CHECK: d2 = sycl::ext::intel::math::erfcinv(d0);
  d2 = erfcinv(d0);
  // CHECK: d2 = sycl::ext::intel::math::erfinv(d0);
  d2 = erfinv(d0);
  // CHECK: d2 = sycl::ext::intel::math::cdfnorm(d0);
  d2 = normcdf(d0);
  // CHECK: d2 = sycl::ext::intel::math::cdfnorminv(d0);
  d2 = normcdfinv(d0);
  // CHECK: d2 = sycl::ext::intel::math::norm(i, (const double *)&d0);
  d2 = norm(i, &d0);
  // CHECK: d2 = sycl::ext::intel::math::norm((int)d1, (const double *)&d0);
  d2 = norm(d1, &d0);
  // CHECK: d2 = sycl::ext::intel::math::rnorm(i, (const double *)&d0);
  d2 = rnorm(i, &d0);
  // CHECK: d2 = sycl::ext::intel::math::rnorm((int)d1, (const double *)&d0);
  d2 = rnorm(d1, &d0);
}

__global__ void kernelFuncFloat(float *deviceArrayFloat) {
  float &f0 = *deviceArrayFloat, &f1 = *(deviceArrayFloat + 1), &f2 = *(deviceArrayFloat + 2);
  int i;

  // Single Precision Mathematical Functions

  // CHECK: f2 = sycl::ext::intel::math::erfcinv(f0);
  f2 = erfcinvf(f0);
  // CHECK: f2 = sycl::ext::intel::math::erfcinv((float)i);
  f2 = erfcinvf(i);
  // CHECK: f2 = sycl::ext::intel::math::erfinv(f0);
  f2 = erfinvf(f0);
  // CHECK: f2 = sycl::ext::intel::math::erfinv((float)i);
  f2 = erfinvf(i);
  // CHECK: f2 = sycl::ext::intel::math::cdfnorm(f0);
  f2 = normcdff(f0);
  // CHECK: f2 = sycl::ext::intel::math::cdfnorm((float)i);
  f2 = normcdff(i);
  // CHECK: f2 = sycl::ext::intel::math::cdfnorminv(f0);
  f2 = normcdfinvf(f0);
  // CHECK: f2 = sycl::ext::intel::math::cdfnorminv((float)i);
  f2 = normcdfinvf(i);
  // CHECK: f2 = sycl::ext::intel::math::norm(i, (const float *)&f0);
  f2 = normf(i, &f0);
  // CHECK: f2 = sycl::ext::intel::math::norm((int)f1, (const float *)&f0);
  f2 = normf(f1, &f0);
  // CHECK: f2 = sycl::ext::intel::math::rnorm(i, (const float *)&f0);
  f2 = rnormf(i, &f0);
  // CHECK: f2 = sycl::ext::intel::math::rnorm((int)f1, (const float *)&f0);
  f2 = rnormf(f1, &f0);
}

__global__ void kernelFuncHalf() {
  __half h, h_1, h_2;
  bool b;

  // Half Arithmetic Functions

  // CHECK: h_2 = sycl::ext::intel::math::hadd_sat(h, h_1);
  h_2 = __hadd_sat(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hfma_sat(h, h_1, h_2);
  h_2 = __hfma_sat(h, h_1, h_2);
  // CHECK: h_2 = sycl::ext::intel::math::hmul_sat(h, h_1);
  h_2 = __hmul_sat(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hsub_sat(h, h_1);
  h_2 = __hsub_sat(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hadd(h, h_1);
  h_2 = __hadd(h, h_1);

  // Half Comparison Functions

  // CHECK: b = sycl::ext::intel::math::hequ(h, h_1);
  b = __hequ(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hgeu(h, h_1);
  b = __hgeu(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hgtu(h, h_1);
  b = __hgtu(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hleu(h, h_1);
  b = __hleu(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hltu(h, h_1);
  b = __hltu(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hneu(h, h_1);
  b = __hneu(h, h_1);
}

__global__ void kernelFuncHalf2() {
  __half2 h2, h2_1, h2_2;
  bool b;

  // Half2 Arithmetic Functions

  // CHECK: h2_2 = sycl::ext::intel::math::hadd2_sat(h2, h2_1);
  h2_2 = __hadd2_sat(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hfma2_sat(h2, h2_1, h2_2);
  h2_2 = __hfma2_sat(h2, h2_1, h2_2);
  // CHECK: h2_2 = sycl::ext::intel::math::hmul2_sat(h2, h2_1);
  h2_2 = __hmul2_sat(h2, h2_1);
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
}

int main() { return 0; }
