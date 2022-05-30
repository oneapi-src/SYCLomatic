// RUN: dpct --format-range=none --out-root %T %s --cuda-include-path="%cuda-path/include" --in-root %S --extra-arg="-I  %S/.."
// RUN: FileCheck --input-file %T/test.dp.cpp --match-full-lines %s

#include "outer/macro_def.h"

void bar() {
  // some work
}

//      CHECK: #define BAR(X)         \
// CHECK-NEXT:   bar();               \
// CHECK-NEXT:   CHECK("cudaGetErrorString not supported"/*cudaGetErrorString(cudaErrorInvalidValue)*/);

#define BAR(X)         \
  bar();               \
  CHECK(cudaGetErrorString(cudaErrorInvalidValue));

void foo() {
  // CHECK: BAR(0)
  BAR(0)
}
