#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          const cuComplex *alpha, cusparseMatDescr_t desc,
          const cuComplex *value, const int *row_ptr, const int *col_idx,
          csrsv2Info_t info, const cuComplex *f, cuComplex *x,
          cusparseSolvePolicy_t policy, void *buffer) {
  // Start
  cusparseCcsrsv2_solve(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, alpha /*const cuComplex **/, desc /*cusparseMatDescr_t*/,
      value /*const cuComplex **/, row_ptr /*const int **/,
      col_idx /*const int **/, info /*csrsv2Info_t*/, f /*const cuComplex **/,
      x /*cuComplex **/, policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
  // End
}
