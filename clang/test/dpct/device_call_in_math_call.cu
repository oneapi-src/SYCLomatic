// RUN: dpct -out-root %T/device_call_in_math_call %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device_call_in_math_call/device_call_in_math_call.dp.cpp

__device__ float g(int width = 32)
{
  warpSize;
  return 1.0;
}
__global__ void f() {
  // CHECK: sycl::fmax((float)(1.0), g(item_ct1, 16));
  fmaxf(1.0, g(16));
}
