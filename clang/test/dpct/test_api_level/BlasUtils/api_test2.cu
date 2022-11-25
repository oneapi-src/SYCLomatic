// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test2_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test2_out

// CHECK: 10

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
