#include "cusparse.h"

void test(const cusparseMatDescr_t desc) {
  // Start
  cusparseFillMode_t uplo =
      cusparseGetMatFillMode(desc /*const cusparseMatDescr_t*/);
  // End
}
