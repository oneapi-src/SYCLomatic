#include "cusparse.h"

void test(const cusparseMatDescr_t desc) {
  // Start
  cusparseMatrixType_t mat_type =
      cusparseGetMatType(desc /*const cusparseMatDescr_t*/);
  // End
}
