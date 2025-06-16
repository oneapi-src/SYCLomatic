#include "cusparse.h"

void test(int64_t rows, int64_t cols, int64_t nnz, void *row_ptr, void *col_idx,
          void *value, cusparseIndexType_t idx_type,
          cusparseIndexBase_t idx_base, cudaDataType value_type) {
  // Start
  cusparseSpMatDescr_t desc;
  cusparseCreateCoo(
      &desc /*cusparseSpMatDescr_t **/, rows /*int64_t*/, cols /*int64_t*/,
      nnz /*int64_t*/, row_ptr /*void **/, col_idx /*void **/, value /*void **/,
      idx_type /*cusparseIndexType_t*/, idx_base /*cusparseIndexBase_t*/,
      value_type /*cudaDataType*/);
  // End
}
