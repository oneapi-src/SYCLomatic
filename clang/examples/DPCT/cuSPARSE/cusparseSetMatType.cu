#include "cusparse.h"

void test(cusparseMatDescr_t desc, cusparseMatrixType_t mat_type) {
  // Start
  cusparseSetMatType(desc, mat_type);
  // End
}
