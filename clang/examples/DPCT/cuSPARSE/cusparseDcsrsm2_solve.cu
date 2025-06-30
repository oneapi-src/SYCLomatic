#include "cusparse.h"

void test(cusparseHandle_t handle, int algo, cusparseOperation_t trans_a,
          cusparseOperation_t trans_b, int m, int nrhs, int nnz,
          const double *alpha, const cusparseMatDescr_t descr,
          const double *value, const int *row_ptr, const int *col_ind,
          double *b, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
          void *buffer) {
  // Start
  cusparseDcsrsm2_solve(
      handle /*cusparseHandle_t*/, algo /*int*/,
      trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
      m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const double **/,
      descr /*const cusparseMatDescr_t*/, value /*const double **/,
      row_ptr /*const int **/, col_ind /*const int **/, b /*double **/,
      ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
      buffer /*void **/);
  // End
}
