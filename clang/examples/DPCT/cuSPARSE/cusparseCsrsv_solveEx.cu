#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m,
          const void *alpha, cudaDataType alpha_type, cusparseMatDescr_t desc,
          const void *value, cudaDataType value_type, const int *row_ptr,
          const int *col_idx, cusparseSolveAnalysisInfo_t info, const void *f,
          cudaDataType f_type, void *x, cudaDataType x_type,
          cudaDataType exec_type) {
  // Start
  cusparseCsrsv_solveEx(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      alpha /*const void **/, alpha_type /*cudaDataType*/,
      desc /*cusparseMatDescr_t*/, value /*const void **/,
      value_type /*cudaDataType*/, row_ptr /*const int **/,
      col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
      f /*const void **/, f_type /*cudaDataType*/, x /*void **/,
      x_type /*cudaDataType*/, exec_type /*cudaDataType*/);
  // End
}
