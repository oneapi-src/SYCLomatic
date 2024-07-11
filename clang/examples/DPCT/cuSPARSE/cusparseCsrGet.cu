#include "cusparse.h"

void test(cusparseSpMatDescr_t desc) {
  // Start
  int64_t rows;
  int64_t cols;
  int64_t nnz;
  void *row_ptr;
  void *col_ind;
  void *value;
  cusparseIndexType_t row_ptr_type;
  cusparseIndexType_t col_ind_type;
  cusparseIndexBase_t base;
  cudaDataType value_type;
  cusparseCsrGet(
      desc /*cusparseSpMatDescr_t*/, &rows /*int64_t **/, &cols /*int64_t **/,
      &nnz /*int64_t **/, &row_ptr /*void ***/, &col_ind /*void ***/,
      &value /*void ***/, &row_ptr_type /*cusparseIndexType_t **/,
      &col_ind_type /*cusparseIndexType_t **/, &base /*cusparseIndexBase_t **/,
      &value_type /*cudaDataType **/);
  // End
}
