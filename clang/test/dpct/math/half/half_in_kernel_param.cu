// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none -out-root %T/math/half/ %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/half/half_in_kernel_param.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

// CHECK: void f(sycl::half a, sycl::half b, sycl::half2 c, sycl::half2 d) {}
__global__ void f(__half a, __half b, __half2 c, __half2 d) {}

int main() {
  // CHECK: f(sycl::half{1}, sycl::half(1.0), sycl::half2{1, 1}, sycl::half2(1.0, 1.0));
  f<<<1, 1>>>(__half{1}, __half(1.0), __half2{1, 1}, __half2(1.0, 1.0));
  return 0;
}
