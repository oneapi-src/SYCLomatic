#include "cusparse.h"

void test(cusparseMatDescr_t desc) {
  // Start
  cusparseDiagType_t diag = cusparseGetMatDiagType(desc /*cusparseMatDescr_t*/);
  // End
}
