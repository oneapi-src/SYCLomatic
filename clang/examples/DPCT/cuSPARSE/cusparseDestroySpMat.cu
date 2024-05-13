#include "cusparse.h"

void test(cusparseSpMatDescr_t desc) {
  // Start
  cusparseDestroySpMat(desc /*cusparseSpMatDescr_t*/);
  // End
}
