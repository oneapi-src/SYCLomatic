// RUN: dpct %S/multi_files.cu -in-root=%S -out-root=%T/multi -analysis-mode -analysis-mode-output-file=%T/multi_report.out

// RUN: echo "// CHECK: %S/multi_files.cu:" > %T/multi_files.check
// RUN: echo "// CHECK: %S/multi_files.h:" >> %T/multi_files.check
// RUN: cat %S/multi_files.check >> %T/multi_files.check

// RUN: FileCheck --match-full-lines --input-file %T/multi_report.out %T/multi_files.check

#include "multi_files.h"

void foo() {
  int *a;
  cudaDeviceGetPCIBusId(nullptr, 0, 0);
  cudaMalloc(&a, sizeof(int) * 4);
}