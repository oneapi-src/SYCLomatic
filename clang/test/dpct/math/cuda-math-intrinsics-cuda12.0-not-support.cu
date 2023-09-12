// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2
// UNSUPPORTED: v12.0, v12.1, v12.2
// RUN: dpct --format-range=none -out-root %T/math/cuda-math-intrinsics-cuda12.0-not-support %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-intrinsics-cuda12.0-not-support/cuda-math-intrinsics-cuda12.0-not-support.dp.cpp --match-full-lines %s

// crt wrapper API
__device__ void foo1() {
  int i;
  float f;
  unsigned int ui;
  long long ll;
  unsigned long long ull;

  // CHECK: i = sycl::mul24(i, i);
  i = mul24(i, i);
  // CHECK: f = dpct::clamp<float>(f, 0.0f, 1.0f);
  f = saturate(f);
}
