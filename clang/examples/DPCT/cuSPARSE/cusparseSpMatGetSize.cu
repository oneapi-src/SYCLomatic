#include "cusparse.h"

void test(cusparseSpMatDescr_t desc) {
  // Start
  int64_t rows;
  int64_t cols;
  int64_t nnz;
  cusparseSpMatGetSize(desc /*cusparseSpMatDescr_t*/, &rows /*int64_t **/,
                       &cols /*int64_t **/, &nnz /*int64_t **/);
  // End
}
