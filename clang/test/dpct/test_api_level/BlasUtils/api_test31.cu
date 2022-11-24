// RUN: dpct --format-range=none    --use-custom-helper=api -out-root %T/BlasUtils/api_test31_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test31_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test31_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test31_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test31_out

// CHECK: 15

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_scal

int main() {
  cublasHandle_t handle;
  void * alpha;
  void * x;

  cublasScalEx(handle, 4, alpha, CUDA_R_32F, x, CUDA_R_32F, 1, CUDA_R_32F);
  return 0;
}
