// RUN: dpct --format-range=none    --use-custom-helper=api -out-root %T/BlasUtils/api_test28_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test28_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test28_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test28_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test28_out

// CHECK: 32

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_herk

int main() {
  cublasHandle_t handle;

  float2 *alpha;
  float *beta;
  float2 *a;
  float2 *b;
  float2 *c;

  cublasCherkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha, a, 3, b, 3, beta, c, 2);
  return 0;
}
