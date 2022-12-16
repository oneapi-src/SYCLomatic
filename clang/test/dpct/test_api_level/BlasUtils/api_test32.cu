// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/BlasUtils/api_test32_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test32_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test32_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test32_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test32_out

// CHECK: 14

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_trsm_batch

int main() {
  cublasHandle_t handle;
  float * alpha;
  const float **a
  float **b;

  cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, alpha, a, 4, b, 4, 2);
  return 0;
}
