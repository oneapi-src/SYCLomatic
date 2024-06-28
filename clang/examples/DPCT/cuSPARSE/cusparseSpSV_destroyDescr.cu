#include "cusparse.h"

void test(cusparseSpSVDescr_t desc) {
  // Start
  cusparseSpSV_destroyDescr(desc /*cusparseSpSVDescr_t*/);
  // End
}
