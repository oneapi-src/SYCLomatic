#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n,
          int k, int nnz, const float *alpha, cusparseMatDescr_t desc,
          const float *value, const int *row_ptr, const int *col_idx,
          const float *B, int ldb, const float *beta, float *C, int ldc) {
  // Start
  cusparseScsrmm(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
                 m /*int*/, n /*int*/, k /*int*/, nnz /*int*/,
                 alpha /*const float **/, desc /*cusparseMatDescr_t*/,
                 value /*const float **/, row_ptr /*const int **/,
                 col_idx /*const int **/, B /*const float **/, ldb /*int*/,
                 beta /*const float **/, C /*float **/, ldc /*int*/);
  // End
}
