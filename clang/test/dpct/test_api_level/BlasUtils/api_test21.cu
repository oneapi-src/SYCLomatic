// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test21_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test21_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test21_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test21_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test21_out

// CHECK: 39

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_syrk

int main() {
  cublasHandle_t handle;
  float *alpha;
  float *beta;
  float *a;
  float *b;
  float *c;

  cublasSsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha, a, 3, b, 3, beta, c, 2);
  return 0;
}
