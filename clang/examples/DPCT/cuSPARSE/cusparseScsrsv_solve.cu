#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m,
          const float *alpha, cusparseMatDescr_t desc, const float *value,
          const int *row_ptr, const int *col_idx,
          cusparseSolveAnalysisInfo_t info, const float *f, float *x) {
  // Start
  cusparseScsrsv_solve(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      alpha /*const float **/, desc /*cusparseMatDescr_t*/,
      value /*const float **/, row_ptr /*const int **/, col_idx /*const int **/,
      info /*cusparseSolveAnalysisInfo_t*/, f /*const float **/, x /*float **/);
  // End
}
