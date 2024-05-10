#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m,
          const double *alpha, cusparseMatDescr_t desc, const double *value,
          const int *row_ptr, const int *col_idx,
          cusparseSolveAnalysisInfo_t info, const double *f, double *x) {
  // Start
  cusparseDcsrsv_solve(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      alpha /*const double **/, desc /*cusparseMatDescr_t*/,
      value /*const double **/, row_ptr /*const int **/,
      col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
      f /*const double **/, x /*double **/);
  // End
}
