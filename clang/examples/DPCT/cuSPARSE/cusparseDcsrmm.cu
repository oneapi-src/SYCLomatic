#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n,
          int k, int nnz, const double *alpha, cusparseMatDescr_t desc,
          const double *value, const int *row_ptr, const int *col_idx,
          const double *B, int ldb, const double *beta, double *C, int ldc) {
  // Start
  cusparseDcsrmm(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
                 m /*int*/, n /*int*/, k /*int*/, nnz /*int*/,
                 alpha /*const double **/, desc /*cusparseMatDescr_t*/,
                 value /*const double **/, row_ptr /*const int **/,
                 col_idx /*const int **/, B /*const double **/, ldb /*int*/,
                 beta /*const double **/, C /*double **/, ldc /*int*/);
  // End
}
