#include "cusparse.h"

void test(cusparseStatus_t status) {
  // Start
  const char *Str = cusparseGetErrorString(status /*cusparseStatus_t*/);
  // End
}
