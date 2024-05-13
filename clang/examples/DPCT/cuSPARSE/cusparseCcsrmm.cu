#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n,
          int k, int nnz, const cuComplex *alpha, cusparseMatDescr_t desc,
          const cuComplex *value, const int *row_ptr, const int *col_idx,
          const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C,
          int ldc) {
  // Start
  cusparseCcsrmm(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
                 m /*int*/, n /*int*/, k /*int*/, nnz /*int*/,
                 alpha /*const cuComplex **/, desc /*cusparseMatDescr_t*/,
                 value /*const cuComplex **/, row_ptr /*const int **/,
                 col_idx /*const int **/, B /*const cuComplex **/, ldb /*int*/,
                 beta /*const cuComplex **/, C /*cuComplex **/, ldc /*int*/);
  // End
}
