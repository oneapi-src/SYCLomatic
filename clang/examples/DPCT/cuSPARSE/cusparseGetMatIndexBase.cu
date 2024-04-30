#include "cusparse.h"

void test(const cusparseMatDescr_t desc) {
  // Start
  cusparseIndexBase_t base = cusparseGetMatIndexBase(desc);
  // End
}
