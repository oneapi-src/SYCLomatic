// RUN: cp %S/MainSourceFiles.yaml %T
// RUN: dpct --out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only 2> %T/output.txt
// RUN: FileCheck --input-file %T/output.txt --match-full-lines %S/output_ref.txt
// RUN: rm -rf %T/*

#include <cuda_runtime.h>

int main() {
  float2 f2;
  return 0;
}