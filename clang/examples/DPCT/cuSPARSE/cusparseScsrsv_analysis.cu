#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          const cusparseMatDescr_t desc, const float *value, const int *row_ptr,
          const int *col_idx, cusparseSolveAnalysisInfo_t info) {
  // Start
  cusparseScsrsv_analysis(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, desc /*const cusparseMatDescr_t*/, value /*const float **/,
      row_ptr /*const int **/, col_idx /*const int **/,
      info /*cusparseSolveAnalysisInfo_t*/);
  // End
}
