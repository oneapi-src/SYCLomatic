#include "cusparse.h"

void test(cusparseConstDnVecDescr_t desc) {
  // Start
  void *value;
  cusparseConstDnVecGetValues(desc /*cusparseConstDnVecDescr_t*/,
                              &value /*void ***/);
  // End
}
