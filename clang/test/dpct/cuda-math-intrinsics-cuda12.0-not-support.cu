// UNSUPPORTED: cuda-12.0
// UNSUPPORTED: v12.0
// RUN: dpct --format-range=none -out-root %T/cuda-math-intrinsics-cuda12.0-not-support %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda-math-intrinsics-cuda12.0-not-support/cuda-math-intrinsics-cuda12.0-not-support.dp.cpp --match-full-lines %s

// crt wrapper API
__device__ void foo1() {
  int i;
  float f;
  unsigned int ui;
  long long ll;
  unsigned long long ull;

  // CHECK: i = sycl::mul24(i, i);
  i = mul24(i, i);
  // CHECK: f = sycl::clamp<float>(f, 0.0f, 1.0f);
  f = saturate(f);
}
