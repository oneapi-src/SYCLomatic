#include "cusparse.h"

void test(cusparseMatDescr_t desc) {
  // Start
  cusparseIndexBase_t base =
      cusparseGetMatIndexBase(desc /*cusparseMatDescr_t*/);
  // End
}
