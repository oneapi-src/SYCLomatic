#include "cusparse.h"

void test(cusparseMatDescr_t desc, cusparseMatrixType_t mat_type) {
  // Start
  cusparseSetMatType(desc /*cusparseMatDescr_t*/,
                     mat_type /*cusparseMatrixType_t*/);
  // End
}
