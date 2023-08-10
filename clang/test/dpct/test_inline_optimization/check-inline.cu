// UNSUPPORTED: system-windows

// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/test.h > %T/test.h
// RUN: cat %S/check-inline.cu > %T/check-inline.cu
// RUN: cat %S/test.cu > %T/test.cu
// RUN: cd %T

// RUN: dpct --format-range=none --optimize-migration -in-root=%T -out-root=%T/test_inline_optimization  -p=%T  --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/test_inline_optimization/check-inline.dp.cpp
// RUN: FileCheck %S/test.h --match-full-lines --input-file %T/test_inline_optimization/test.h
// RUN: FileCheck %S/test.cu --match-full-lines --input-file %T/test_inline_optimization/test.dp.cpp

#include "test.h"
// CHECK: inline int test() {
__host__ __device__ int test() {
  return 5;
}

__global__ void kernel() {
  test();
  test1();
  test2();
}

int main() {
  kernel<<<1, 1>>>();
  return 0;
}
