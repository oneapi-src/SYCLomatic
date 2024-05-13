#include "cusparse.h"

void test(cusparseMatDescr_t desc) {
  // Start
  cusparseDestroyMatDescr(desc /*cusparseMatDescr_t*/);
  // End
}
