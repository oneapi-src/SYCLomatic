#include "cusparse.h"

void test(cusparseSpMatDescr_t desc) {
  // Start
  cusparseIndexBase_t base;
  cusparseSpMatGetIndexBase(desc /*cusparseSpMatDescr_t*/,
                            &base /*cusparseIndexBase_t **/);
  // End
}
