#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n,
          int nnz, const cuDoubleComplex *alpha, cusparseMatDescr_t desc,
          const cuDoubleComplex *value, const int *row_ptr, const int *col_idx,
          const cuDoubleComplex *x, const cuDoubleComplex *beta,
          cuDoubleComplex *y) {
  // Start
  cusparseZcsrmv_mp(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
                    m /*int*/, n /*int*/, nnz /*int*/,
                    alpha /*const cuDoubleComplex **/,
                    desc /*cusparseMatDescr_t*/,
                    value /*const cuDoubleComplex **/, row_ptr /*const int **/,
                    col_idx /*const int **/, x /*const cuDoubleComplex **/,
                    beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/);
  // End
}
