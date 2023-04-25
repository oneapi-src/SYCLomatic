// RUN: dpct --format-range=none    --use-custom-helper=api -out-root %T/BlasUtils/api_test24_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test24_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test24_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test24_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test24_out

// CHECK: 27

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_dotc

int main() {
  cublasHandle_t handle;
  const void *x;
  const void *y;
  void *res;

  cublasDotcEx(handle, 4, x, CUDA_R_32F, 1, y, CUDA_R_32F, 1, res, CUDA_R_32F, CUDA_R_32F);
  return 0;
}
