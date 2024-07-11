#include "cusparse.h"

void test(cusparseSpSMDescr_t desc) {
  // Start
  cusparseSpSM_destroyDescr(desc /*cusparseSpSMDescr_t*/);
  // End
}
