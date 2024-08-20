// RUN: dpct --out-root %T %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/test.dp.cpp
// RUN: FileCheck %S/cuda.h --match-full-lines --input-file %T/cuda.h
// RUN: %if build_lit %{icpx -c -fsycl %T/test.dp.cpp -o %T/test.dp.o %}

#include "cuda.h"

int main() {
  // CHECK: sycl::float2 f2;
  float2 f2;
  return 0;
}
