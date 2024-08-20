#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          const double *alpha, cusparseMatDescr_t desc, const double *value,
          const int *row_ptr, const int *col_idx, csrsv2Info_t info,
          const double *f, double *x, cusparseSolvePolicy_t policy,
          void *buffer) {
  // Start
  cusparseDcsrsv2_solve(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, alpha /*const double **/, desc /*cusparseMatDescr_t*/,
      value /*const double **/, row_ptr /*const int **/,
      col_idx /*const int **/, info /*csrsv2Info_t*/, f /*const double **/,
      x /*double **/, policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
  // End
}
