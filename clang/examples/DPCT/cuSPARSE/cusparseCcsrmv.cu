#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n,
          int nnz, const cuComplex *alpha, cusparseMatDescr_t desc,
          const cuComplex *value, const int *row_ptr, const int *col_idx,
          const cuComplex *x, const cuComplex *beta, cuComplex *y) {
  // Start
  cusparseCcsrmv(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
                 m /*int*/, n /*int*/, nnz /*int*/, alpha /*const cuComplex **/,
                 desc /*cusparseMatDescr_t*/, value /*const cuComplex **/,
                 row_ptr /*const int **/, col_idx /*const int **/,
                 x /*const cuComplex **/, beta /*const cuComplex **/,
                 y /*cuComplex **/);
  // End
}
