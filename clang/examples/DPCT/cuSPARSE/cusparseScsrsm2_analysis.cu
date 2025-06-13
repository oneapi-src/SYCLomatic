#include "cusparse.h"

void test(cusparseHandle_t handle, int algo, cusparseOperation_t trans_a,
          cusparseOperation_t trans_b, int m, int nrhs, int nnz,
          const float *alpha, const cusparseMatDescr_t descr,
          const float *value, const int *row_ptr, const int *col_ind,
          const float *b, int ldb, csrsm2Info_t info,
          cusparseSolvePolicy_t policy, void *buffer) {
  // Start
  cusparseScsrsm2_analysis(
      handle /*cusparseHandle_t*/, algo /*int*/,
      trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
      m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const float **/,
      descr /*const cusparseMatDescr_t*/, value /*const float **/,
      row_ptr /*const int **/, col_ind /*const int **/, b /*const float **/,
      ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
      buffer /*void **/);
  // End
}
