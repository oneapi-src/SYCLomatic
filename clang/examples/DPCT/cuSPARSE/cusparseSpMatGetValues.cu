#include "cusparse.h"

void test(cusparseSpMatDescr_t desc) {
  // Start
  void *value;
  cusparseSpMatGetValues(desc /*cusparseSpMatDescr_t*/, &value /*void ***/);
  // End
}
