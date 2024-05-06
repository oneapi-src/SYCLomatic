#include "cusparse.h"

void test(cusparseHandle_t handle, cusparseAlgMode_t algo,
          cusparseOperation_t trans, int m, int n, int nnz, const void *alpha,
          cudaDataType alpha_type, const cusparseMatDescr_t desc,
          const void *value, cudaDataType value_type, const int *row_ptr,
          const int *col_idx, const void *x, cudaDataType x_type,
          const void *beta, cudaDataType beta_type, void *y,
          cudaDataType y_type, cudaDataType exec_type, void *buffer) {
  // Start
  cusparseCsrmvEx(
      handle /*cusparseHandle_t*/, algo /*cusparseAlgMode_t*/,
      trans /*cusparseOperation_t*/, m /*int*/, n /*int*/, nnz /*int*/,
      alpha /*const void **/, alpha_type /*cudaDataType*/,
      desc /*const cusparseMatDescr_t*/, value /*const void **/,
      value_type /*cudaDataType*/, row_ptr /*const int **/,
      col_idx /*const int **/, x /*const void **/, x_type /*cudaDataType*/,
      beta /*const void **/, beta_type /*cudaDataType*/, y /*void **/,
      y_type /*cudaDataType*/, exec_type /*cudaDataType*/, buffer /*void **/);
  // End
}
