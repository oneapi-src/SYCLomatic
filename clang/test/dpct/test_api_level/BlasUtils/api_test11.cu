// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test11_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test11_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test11_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test11_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test11_out

// CHECK: 24

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_axpy

int main() {
  cublasHandle_t handle;
  const void *alpha;
  const void *x;
  void *y;

  cublasAxpyEx(handle, 4, alpha, CUDA_R_32F, x, CUDA_R_32F, 1, y, CUDA_R_32F, 1, CUDA_R_32F);
  return 0;
}
