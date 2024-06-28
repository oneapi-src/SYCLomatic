#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m,
          const cuComplex *alpha, cusparseMatDescr_t desc,
          const cuComplex *value, const int *row_ptr, const int *col_idx,
          cusparseSolveAnalysisInfo_t info, const cuComplex *f, cuComplex *x) {
  // Start
  cusparseCcsrsv_solve(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      alpha /*const cuComplex **/, desc /*cusparseMatDescr_t*/,
      value /*const cuComplex **/, row_ptr /*const int **/,
      col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
      f /*const cuComplex **/, x /*cuComplex **/);
  // End
}
