#include "cusparse.h"

void test(cusparseHandle_t handle, int m, int n, int nnz,
          const cuDoubleComplex *csr_value, const int *row_ptr,
          const int *col_idx, cuDoubleComplex *csc_value, int *row_ind,
          int *col_ptr, cusparseAction_t act, cusparseIndexBase_t base) {
  // Start
  cusparseZcsr2csc(handle /*cusparseHandle_t*/, m /*int*/, n /*int*/,
                   nnz /*int*/, csr_value /*const cuDoubleComplex **/,
                   row_ptr /*const int **/, col_idx /*const int **/,
                   csc_value /*cuDoubleComplex **/, row_ind /*int **/,
                   col_ptr /*int **/, act /*cusparseAction_t*/,
                   base /*cusparseIndexBase_t*/);
  // End
}
