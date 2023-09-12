// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/math/cuda-math-extension-cuda11-after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-extension-cuda11-after/cuda-math-extension-cuda11-after.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/ext/intel/math.hpp>
#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncHalf(__half *deviceArrayHalf) {
  __half h, h_1, h_2;
  __half2 h2, h2_1, h2_2;

  // Half Arithmetic Functions

  // CHECK: h_2 = sycl::ext::intel::math::hadd(h, h_1);
  h_2 = __hadd_rn(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hfma_relu(h, h_1, h_2);
  h_2 = __hfma_relu(h, h_1, h_2);
  // CHECK: h_2 = sycl::ext::intel::math::hmul(h, h_1);
  h_2 = __hmul_rn(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hsub(h, h_1);
  h_2 = __hsub_rn(h, h_1);

  // Half2 Arithmetic Functions

  // CHECK: h2_2 = sycl::ext::intel::math::hadd2(h2, h2_1);
  h2_2 = __hadd2_rn(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hcmadd(h2, h2_1, h2_2);
  h2_2 = __hcmadd(h2, h2_1, h2_2);
  // CHECK: h2_2 = sycl::ext::intel::math::hfma2_relu(h2, h2_1, h2_2);
  h2_2 = __hfma2_relu(h2, h2_1, h2_2);
  // CHECK: h2_2 = sycl::ext::intel::math::hmul2(h2, h2_1);
  h2_2 = __hmul2_rn(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hsub2(h2, h2_1);
  h2_2 = __hsub2_rn(h2, h2_1);

  // Half Comparison Functions

  // CHECK: h_2 = sycl::ext::intel::math::hmax(h, h_1);
  h_2 = __hmax(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hmax_nan(h, h_1);
  h_2 = __hmax_nan(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hmin(h, h_1);
  h_2 = __hmin(h, h_1);
  // CHECK: h_2 = sycl::ext::intel::math::hmin_nan(h, h_1);
  h_2 = __hmin_nan(h, h_1);

  // Half2 Comparison Functions

  // CHECK: h2_2 = sycl::ext::intel::math::hmax2(h2, h2_1);
  h2_2 = __hmax2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hmax2_nan(h2, h2_1);
  h2_2 = __hmax2_nan(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hmin2(h2, h2_1);
  h2_2 = __hmin2(h2, h2_1);
  // CHECK: h2_2 = sycl::ext::intel::math::hmin2_nan(h2, h2_1);
  h2_2 = __hmin2_nan(h2, h2_1);
}

int main() { return 0; }
