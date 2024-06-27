#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m,
          const cuDoubleComplex *alpha, cusparseMatDescr_t desc,
          const cuDoubleComplex *value, const int *row_ptr, const int *col_idx,
          cusparseSolveAnalysisInfo_t info, const cuDoubleComplex *f,
          cuDoubleComplex *x) {
  // Start
  cusparseZcsrsv_solve(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      alpha /*const cuDoubleComplex **/, desc /*cusparseMatDescr_t*/,
      value /*const cuDoubleComplex **/, row_ptr /*const int **/,
      col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
      f /*const cuDoubleComplex **/, x /*cuDoubleComplex **/);
  // End
}
