// RUN: dpct --format-range=none --out-root %T %s --cuda-include-path="%cuda-path/include" --in-root %S --extra-arg="-I  %S/.."
// RUN: FileCheck --input-file %T/test.dp.cpp --match-full-lines %s

#include "outer/macro_def.h"
#include "cuda_runtime.h"

void foo() {}

// CHECK: #define MACRO_B \
// CHECK-NEXT: foo();\
// CHECK-NEXT: MACRO("cudaGetErrorString not supported"/*cudaGetErrorString(cudaErrorInvalidValue)*/);
#define MACRO_B \
foo();\
MACRO(cudaGetErrorString(cudaErrorInvalidValue));

int main() {
  // CHECK: MACRO_B
  MACRO_B
  return 0;
}
