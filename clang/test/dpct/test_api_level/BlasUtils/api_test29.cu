// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/BlasUtils/api_test29_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test29_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test29_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test29_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test29_out

// CHECK: 27

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_nrm2

int main() {
  cublasHandle_t handle;
  void * x;
  void * res;

  cublasNrm2Ex(handle, 4, x, CUDA_R_32F, 1, res, CUDA_R_32F, CUDA_R_32F);
  return 0;
}
