#include "cusparse.h"

void test(cusparseHandlet handle) {
  // Start
  cusparsePointerMode_t mode;
  cusparseGetPointerMode(handle, &mode);
  // End
}
