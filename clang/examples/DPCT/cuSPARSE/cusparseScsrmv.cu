#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n,
          int nnz, const float *alpha, cusparseMatDescr_t desc,
          const float *value, const int *row_ptr, const int *col_idx,
          const float *x, const float *beta, float *y) {
  // Start
  cusparseScsrmv(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
                 m /*int*/, n /*int*/, nnz /*int*/, alpha /*const float **/,
                 desc /*cusparseMatDescr_t*/, value /*const float **/,
                 row_ptr /*const int **/, col_idx /*const int **/,
                 x /*const float **/, beta /*const float **/, y /*float **/);
  // End
}
