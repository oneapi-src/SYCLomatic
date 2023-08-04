// RUN: dpct --format-range=none --optimize-migration -out-root %T/test_inline_optimization %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/test_inline_optimization/check-inline.dp.cpp
// RUN: FileCheck %S/test.h --match-full-lines --input-file %T/test_inline_optimization/test.h

#include "test.h"
// CHECK: int test() {
__host__ __device__ int test() {
  return 5;
}

__global__ void kernel() {
  test();
  test1();
}

int main() {
  kernel<<<1, 1>>>();
  return 0;
}
