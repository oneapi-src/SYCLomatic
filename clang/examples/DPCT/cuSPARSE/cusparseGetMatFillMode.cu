#include "cusparse.h"

void test(cusparseMatDescr_t desc) {
  // Start
  cusparseFillMode_t uplo = cusparseGetMatFillMode(desc /*cusparseMatDescr_t*/);
  // End
}
