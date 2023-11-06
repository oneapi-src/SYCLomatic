// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// RUN: dpct --format-range=none --use-experimental-features=bfloat16_math_functions -out-root %T/math/bfloat16/bfloat16_cuda12_after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/math/bfloat16/bfloat16_cuda12_after/bfloat16_cuda12_after.dp.cpp

#include "cuda_bf16.h"

__global__ void kernelFuncBfloat162Comparison() {
  // CHECK: sycl::marray<sycl::ext::oneapi::bfloat16, 2> bf162, bf162_1, bf162_2;
  __nv_bfloat162 bf162, bf162_1, bf162_2;
  unsigned u;

  // Half2 Comparison Functions

  // CHECK: u = dpct::compare_mask(bf162_1, bf162_2, std::equal_to<>());
  u = __heq2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::unordered_compare_mask(bf162_1, bf162_2, std::equal_to<>());
  u = __hequ2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::compare_mask(bf162_1, bf162_2, std::greater_equal<>());
  u = __hge2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::unordered_compare_mask(bf162_1, bf162_2, std::greater_equal<>());
  u = __hgeu2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::compare_mask(bf162_1, bf162_2, std::greater<>());
  u = __hgt2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::unordered_compare_mask(bf162_1, bf162_2, std::greater<>());
  u = __hgtu2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::compare_mask(bf162_1, bf162_2, std::less_equal<>());
  u = __hle2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::unordered_compare_mask(bf162_1, bf162_2, std::less_equal<>());
  u = __hleu2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::compare_mask(bf162_1, bf162_2, std::less<>());
  u = __hlt2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::unordered_compare_mask(bf162_1, bf162_2, std::less<>());
  u = __hltu2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::compare_mask(bf162_1, bf162_2, std::not_equal_to<>());
  u = __hne2_mask(bf162_1, bf162_2);
  // CHECK: u = dpct::unordered_compare_mask(bf162_1, bf162_2, std::not_equal_to<>());
  u = __hneu2_mask(bf162_1, bf162_2);
}

int main() { return 0; }
