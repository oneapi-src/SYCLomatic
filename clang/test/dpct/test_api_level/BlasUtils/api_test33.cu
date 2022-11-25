// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/BlasUtils/api_test33_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test33_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test33_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test33_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test33_out

// CHECK: 32

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_trmm

int main() {
  cublasHandle_t handle;
  float * alpha;
  float *a, *b, *c;

  cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, alpha, a, 4, b, 4, c, 2);
  return 0;
}
