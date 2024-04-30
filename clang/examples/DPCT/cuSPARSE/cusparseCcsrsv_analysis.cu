#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          const void *alpha, const cusparseMatDescr_t desc,
          const cuComplex *value, const int *row_ptr, const int *col_idx,
          cusparseSolveAnalysisInfo_t info) {
  // Start
  cusparseCcsrsv_analysis(handle, trans, m, nnz, desc, value, row_ptr, col_idx,
                          info);
  // End
}
