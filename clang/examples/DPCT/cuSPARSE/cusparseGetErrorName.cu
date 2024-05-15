#include "cusparse.h"

void test(cusparseStatus_t status) {
  // Start
  const char *Name = cusparseGetErrorName(status /*cusparseStatus_t*/);
  // End
}
