#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          cusparseMatDescr_t desc, const cuDoubleComplex *value,
          const int *row_ptr, const int *col_idx,
          cusparseSolveAnalysisInfo_t info) {
  // Start
  cusparseZcsrsv_analysis(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, desc /*cusparseMatDescr_t*/,
      value /*const cuDoubleComplex **/, row_ptr /*const int **/,
      col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/);
  // End
}
