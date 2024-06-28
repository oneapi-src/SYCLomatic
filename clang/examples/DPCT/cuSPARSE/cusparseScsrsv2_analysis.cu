#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          cusparseMatDescr_t desc, const float *value, const int *row_ptr,
          const int *col_idx, csrsv2Info_t info, cusparseSolvePolicy_t policy,
          void *buffer) {
  // Start
  cusparseScsrsv2_analysis(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*const float **/,
      row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
      policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
  // End
}
