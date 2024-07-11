#include "cusparse.h"

void test(cusparseSpMatDescr_t desc, void *value) {
  // Start
  cusparseSpMatSetValues(desc /*cusparseSpMatDescr_t*/, value /*void **/);
  // End
}
