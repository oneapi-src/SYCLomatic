// RUN: dpct --format-range=none -out-root %T/c_feature_file %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err --extra-arg="-xc"
// RUN: FileCheck %s --match-full-lines --input-file %T/c_feature_file/c_feature_file.c.dp.cpp

//CHECK:#include <sycl/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
#include "cuda_runtime.h"
#include <stdio.h>

void func(int N, double re[][1<<N]) {
  printf("Hello from bindArraysToStackComplexMatrixN\n");
}

int main(int argc, char** argv) {
  const int N = 4;
  double a[1<<(N)][1<<(N)];

  func(N, a);
  return 0;
}