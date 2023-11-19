// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none -out-root %T/compat_with_clang_2 %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err --extra-arg="-std=c++14"
// RUN: FileCheck %s --match-full-lines --input-file %T/compat_with_clang_2/compat_with_clang_2.dp.cpp

#include <cuda_runtime.h>

const float FOUR_EIGHT_ZERO = 480.0f;

// CHECK: void foo() {
// CHECK-NEXT:   float max = -10000.f;
// CHECK-NEXT:   float res = std::max(max / FOUR_EIGHT_ZERO, 1.0f / 32.f);
// CHECK-NEXT: }
__global__ void foo() {
  float max = -10000.f;
  float res = std::max(max / FOUR_EIGHT_ZERO, 1.0f / 32.f);
}
