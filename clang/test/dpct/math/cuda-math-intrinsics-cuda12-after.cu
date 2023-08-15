// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// RUN: dpct --format-range=none -out-root %T/math/cuda-math-intrinsics-cuda12-after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-intrinsics-cuda12-after/cuda-math-intrinsics-cuda12-after.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncHalf(__half *deviceArrayHalf) {
  __half h, h_1, h_2;
  __half2 h2, h2_1, h2_2;

  // Half2 Comparison Functions

  // CHECK: h2_2 = dpct::compare_mask(h2, h2_1, std::equal_to<>());
  h2_2 = __heq2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::unordered_compare_mask(h2, h2_1, std::equal_to<>());
  h2_2 = __hequ2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::compare_mask(h2, h2_1, std::greater_equal<>());
  h2_2 = __hge2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::unordered_compare_mask(h2, h2_1, std::greater_equal<>());
  h2_2 = __hgeu2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::compare_mask(h2, h2_1, std::greater<>());
  h2_2 = __hgt2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::unordered_compare_mask(h2, h2_1, std::greater<>());
  h2_2 = __hgtu2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::compare_mask(h2, h2_1, std::less_equal<>());
  h2_2 = __hle2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::unordered_compare_mask(h2, h2_1, std::less_equal<>());
  h2_2 = __hleu2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::compare_mask(h2, h2_1, std::less<>());
  h2_2 = __hlt2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::unordered_compare_mask(h2, h2_1, std::less<>());
  h2_2 = __hltu2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::compare_mask(h2, h2_1, std::not_equal_to<>());
  h2_2 = __hne2_mask(h2, h2_1);
  // CHECK: h2_2 = dpct::unordered_compare_mask(h2, h2_1, std::not_equal_to<>());
  h2_2 = __hneu2_mask(h2, h2_1);
}

int main() { return 0; }
