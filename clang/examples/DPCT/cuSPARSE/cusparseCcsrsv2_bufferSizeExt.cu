#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseOperation_t trans, int m, int nnz,
          cusparseMatDescr_t desc, cuComplex *value, const int *row_ptr,
          const int *con_ind, csrsv2Info_t info) {
  // Start
  size_t buffer_size;
  cusparseCcsrsv2_bufferSizeExt(
      handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
      nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*cuComplex **/,
      row_ptr /*const int **/, con_ind /*const int **/, info /*csrsv2Info_t*/,
      &buffer_size /*size_t **/);
  // End
}
