#include "cusparse.h"

void test(cusparseSpGEMMDescr_t desc) {
  // Start
  cusparseSpGEMM_destroyDescr(desc /*cusparseSpGEMMDescr_t*/);
  // End
}
