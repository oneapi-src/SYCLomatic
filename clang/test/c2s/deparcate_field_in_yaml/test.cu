// RUN: cp %S/MainSourceFiles.yaml %T
// RUN: c2s --out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only 2> %T/output.txt
// RUN: FileCheck --input-file %T/output.txt --match-full-lines %S/output_ref.txt
// RUN: grep -n "DpctVersion\|USMLevel" %T/MainSourceFiles.yaml | wc -l > %T/wc_output.txt || true
// RUN: FileCheck --input-file %T/wc_output.txt --match-full-lines %s
// RUN: rm -rf %T/*

// CHECK: 0

#include <cuda_runtime.h>

int main() {
  float2 f2;
  return 0;
}