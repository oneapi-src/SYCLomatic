#include "cusparse.h"

void test(cusparseConstDnVecDescr_t desc) {
  // Start
  const void *value;
  cusparseConstDnVecGetValues(desc /*cusparseConstDnVecDescr_t*/,
                              &value /*const void ***/);
  // End
}
