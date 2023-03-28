// REQUIRES: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// REQUIRES:  cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test2_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test2_out

// CHECK: 11

// TEST_FEATURE: BlasUtils_get_transpose

#include "cusparse_v2.h"

int main() {
  cusparseHandle_t handle;
  cusparseMatDescr_t descrA;
  double *alpha;
  const double* csrValA;
  const int* csrRowPtrA;
  const int* csrColIndA;
  const double* x;
  double beta;
  double* y;
  int aaaaa = 0;
  cusparseDcsrmv(handle, (cusparseOperation_t)aaaaa, 2, 2, 2, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
  return 0;
}
