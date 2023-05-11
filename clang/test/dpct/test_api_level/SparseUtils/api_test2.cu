// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1
// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/SparseUtils/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/SparseUtils/api_test2_out/MainSourceFiles.yaml | wc -l > %T/SparseUtils/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/SparseUtils/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/SparseUtils/api_test2_out

// CHECK: 11
// TEST_FEATURE: SparseUtils_csrmv

#include "cusparse.h"

int main() {
  cusparseHandle_t handle;
  cusparseOperation_t trans;
  int m;
  int n;
  int nnz;
  float *alpha;
  cusparseMatDescr_t descr;
  float *csrVal;
  int *csrRowPtr;
  int *csrColInd;
  float *x;
  float *beta;
  float *y;
  cusparseScsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr,
                 csrColInd, x, beta, y);
  return 0;
}
