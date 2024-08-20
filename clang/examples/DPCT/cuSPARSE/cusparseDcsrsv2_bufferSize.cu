#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          cusparseMatDescr_t desc, double *value, const int *row_ptr,
          const int *col_idx, csrsv2Info_t info) {
  // Start
  int buffer_size_in_bytes;
  cusparseDcsrsv2_bufferSize(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*double **/,
      row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
      &buffer_size_in_bytes /*int **/);
  // End
}
