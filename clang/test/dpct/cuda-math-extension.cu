// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/cuda-math-extension %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda-math-extension/cuda-math-extension.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

using namespace std;

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

  // CHECK: h_2 = sycl::ext::intel::math::hmul(h, h_1);
  h_2 = __hmul(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hsub(h, h_1);
  h_2 = __hsub(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hneg(h_1);
  h_2 = __hneg(h_1);


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
  // CHECK: b = sycl::ext::intel::math::heq(h, h_1);
  b = __heq(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hne(h, h_1);
  b = __hne(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hge(h, h_1);
  b = __hge(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hgt(h, h_1);
  b = __hgt(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hle(h, h_1);
  b = __hle(h, h_1);
  // CHECK: b = sycl::ext::intel::math::hlt(h, h_1);
  b = __hlt(h, h_1);
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
  // CHECK: h2_2 = sycl::ext::intel::math::hadd2(h2, h2_1);
  h2_2 = __hadd2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hmul2(h2, h2_1);
  h2_2 = __hmul2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hsub2(h2, h2_1);
  h2_2 = __hsub2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hneg2(h2_1);
  h2_2 = __hneg2(h2_1);
}

int main() { return 0; }
