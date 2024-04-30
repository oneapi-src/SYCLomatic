#include "cusparse.h"

void test(cusparseMatDescr_t desc, cusparseDiagType_t diag) {
  // Start
  cusparseSetMatDiagType(desc /*cusparseMatDescr_t*/,
                         diag /*cusparseDiagType_t*/);
  // End
}
