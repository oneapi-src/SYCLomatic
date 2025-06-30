#include "cusparse.h"

void test(cusparseHandle_t handle, int algo, cusparseOperation_t trans_a,
          cusparseOperation_t trans_b, int m, int nrhs, int nnz,
          const float *alpha, const cusparseMatDescr_t descr,
          const float *value, const int *row_ptr, const int *col_ind,
          const float *b, int ldb, csrsm2Info_t info,
          cusparseSolvePolicy_t policy, size_t *buffer_size) {
  // Start
  cusparseScsrsm2_bufferSizeExt(
      handle /*cusparseHandle_t*/, algo /*int*/,
      trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
      m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const float **/,
      descr /*const cusparseMatDescr_t*/, value /*const float **/,
      row_ptr /*const int **/, col_ind /*const int **/, b /*const float **/,
      ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
      buffer_size /*size_t **/);
  // End
}
