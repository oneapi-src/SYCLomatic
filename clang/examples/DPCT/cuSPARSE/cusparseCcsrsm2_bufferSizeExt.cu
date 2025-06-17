#include "cusparse.h"

void test(cusparseHandle_t handle, int algo, cusparseOperation_t trans_a,
          cusparseOperation_t trans_b, int m, int nrhs, int nnz,
          const cuComplex *alpha, const cusparseMatDescr_t descr,
          const cuComplex *value, const int *row_ptr, const int *col_ind,
          const cuComplex *b, int ldb, csrsm2Info_t info,
          cusparseSolvePolicy_t policy, size_t *buffer_size) {
  // Start
  cusparseCcsrsm2_bufferSizeExt(
      handle /*cusparseHandle_t*/, algo /*int*/,
      trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
      m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const cuComplex **/,
      descr /*const cusparseMatDescr_t*/, value /*const cuComplex **/,
      row_ptr /*const int **/, col_ind /*const int **/, b /*const cuComplex **/,
      ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
      buffer_size /*size_t **/);
  // End
}
