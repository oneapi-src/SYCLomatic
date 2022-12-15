// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test12_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test12_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test12_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test12_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test12_out

// CHECK: 18

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_dot

int main() {
  cublasHandle_t handle;
  const void *x;
  const void *y;
  void *res;

  cublasDotEx(handle, 4, x, CUDA_R_32F, 1, y, CUDA_R_32F, 1, res, CUDA_R_32F, CUDA_R_32F);
  return 0;
}
