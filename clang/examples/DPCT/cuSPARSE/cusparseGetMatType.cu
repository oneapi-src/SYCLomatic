#include "cusparse.h"

void test(cusparseMatDescr_t desc) {
  // Start
  cusparseMatrixType_t mat_type =
      cusparseGetMatType(desc /*cusparseMatDescr_t*/);
  // End
}
