#include "cusparse.h"

void test(const cusparseMatDescr_t desc) {
  // Start
  cusparseDiagType_t diag =
      cusparseGetMatDiagType(desc /*const cusparseMatDescr_t*/);
  // End
}
