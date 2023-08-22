// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none -out-root %T/math/cuda-math-habs %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-habs/cuda-math-habs.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <cuda_fp16.h>

__global__ void test() {
  __half h, h_2;
  __half2 h2, h2_2;

  // CHECK: h_2 = sycl::fabs(h);
  h_2 = __habs(h);

  // CHECK: h2_2 = sycl::fabs(h2);
  h2_2 = __habs2(h2);
}
