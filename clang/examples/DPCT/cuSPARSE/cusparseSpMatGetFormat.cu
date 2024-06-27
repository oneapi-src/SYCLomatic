#include "cusparse.h"

void test(cusparseSpMatDescr_t desc) {
  // Start
  cusparseFormat_t format;
  cusparseSpMatGetFormat(desc /*cusparseSpMatDescr_t*/,
                         &format /*cusparseFormat_t **/);
  // End
}
