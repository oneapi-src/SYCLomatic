// RUN: dpct -in-root=%S -analysis-mode --cuda-include-path="%cuda-path/include" -analysis-mode-output-file=%T/single_file.out %S/single_file.cu

// RUN: echo "// CHECK: {{.*}}single_file.cu:" > %T/single_file.check
// RUN: cat %S/single_file.check >> %T/single_file.check

// RUN: FileCheck --match-full-lines --input-file %T/single_file.out %T/single_file.check 

#include "cudnn.h"

void foo() {
  int *a;
  cudnnReduceTensorIndices_t m;
  cudaDeviceGetPCIBusId(nullptr, 0, 0);
  cudaDeviceGetPCIBusId(nullptr, 0, 0);
  cudaMalloc(&a, sizeof(int) * 4);
}