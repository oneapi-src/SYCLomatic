#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          cusparseMatDescr_t desc, const void *value, cudaDataType value_type,
          const int *row_ptr, const int *col_idx,
          cusparseSolveAnalysisInfo_t info, cudaDataType exec_type) {
  // Start
  cusparseCsrsv_analysisEx(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*const void **/,
      value_type /*cudaDataType*/, row_ptr /*const int **/,
      col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
      exec_type /*cudaDataType*/);
  // End
}
