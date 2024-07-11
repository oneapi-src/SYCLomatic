#include "cusparse.h"

void test(cusparseMatDescr_t desc, cusparseFillMode_t uplo) {
  // Start
  cusparseSetMatFillMode(desc /*cusparseMatDescr_t*/,
                         uplo /*cusparseFillMode_t*/);
  // End
}
