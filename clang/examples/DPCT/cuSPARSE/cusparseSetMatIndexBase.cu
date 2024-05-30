#include "cusparse.h"

void test(cusparseMatDescr_t desc, cusparseIndexBase_t base) {
  // Start
  cusparseSetMatIndexBase(desc /*cusparseMatDescr_t*/,
                          base /*cusparseIndexBase_t*/);
  // End
}
