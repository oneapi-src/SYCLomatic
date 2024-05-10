#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int n,
          int nnz, const double *alpha, cusparseMatDescr_t desc,
          const double *value, const int *row_ptr, const int *col_idx,
          const double *x, const double *beta, double *y) {
  // Start
  cusparseDcsrmv_mp(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
                    m /*int*/, n /*int*/, nnz /*int*/, alpha /*const double **/,
                    desc /*cusparseMatDescr_t*/, value /*const double **/,
                    row_ptr /*const int **/, col_idx /*const int **/,
                    x /*const double **/, beta /*const double **/,
                    y /*double **/);
  // End
}
