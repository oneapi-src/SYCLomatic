#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans_a,
          cusparseOperation_t trans_b, int m, int n, int k, int nnz,
          const cuDoubleComplex *alpha, cusparseMatDescr_t desc,
          const cuDoubleComplex *value, const int *row_ptr, const int *col_idx,
          const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
          cuDoubleComplex *C, int ldc) {
  // Start
  cusparseZcsrmm2(handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
                  trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/,
                  k /*int*/, nnz /*int*/, alpha /*const cuDoubleComplex **/,
                  desc /*cusparseMatDescr_t*/,
                  value /*const cuDoubleComplex **/, row_ptr /*const int **/,
                  col_idx /*const int **/, B /*const cuDoubleComplex **/,
                  ldb /*int*/, beta /*const cuDoubleComplex **/,
                  C /*cuDoubleComplex **/, ldc /*int*/);
  // End
}
