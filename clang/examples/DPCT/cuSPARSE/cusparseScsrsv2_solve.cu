#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          const float *alpha, cusparseMatDescr_t desc, const float *value,
          const int *row_ptr, const int *col_idx, csrsv2Info_t info,
          const float *f, float *x, cusparseSolvePolicy_t policy,
          void *buffer) {
  // Start
  cusparseScsrsv2_solve(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, alpha /*const float **/, desc /*cusparseMatDescr_t*/,
      value /*const float **/, row_ptr /*const int **/, col_idx /*const int **/,
      info /*csrsv2Info_t*/, f /*const float **/, x /*float **/,
      policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
  // End
}
