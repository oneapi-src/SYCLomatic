// RUN: dpct --format-range=none --optimize-migration -out-root %T/test_inline_optimization %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/test_inline_optimization/check-inline.dp.cpp
// RUN: FileCheck %S/test.h --match-full-lines --input-file %T/test_inline_optimization/test.h

#include "test.h"
// CHECK: int test() {
__host__ __device__ int test() {
  return 5;
}

int main() {
  test();
  test1();
  return 0;
}
