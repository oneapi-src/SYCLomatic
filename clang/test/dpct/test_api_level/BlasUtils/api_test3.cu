// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test3_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test3_out

// CHECK: 9

// TEST_FEATURE: BlasUtils_get_value

#include "cublas_v2.h"

int main() {
  cublasHandle_t handle;
  float *alpha_S, *x_S, *result_S;
  int incx, incy;
  cublasSaxpy(handle, 10, alpha_S, x_S, incx, result_S, incy);
  return 0;
}
