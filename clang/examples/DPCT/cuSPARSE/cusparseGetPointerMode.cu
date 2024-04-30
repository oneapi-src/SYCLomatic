#include "cusparse.h"

void test(cusparseHandle_t handle) {
  // Start
  cusparsePointerMode_t mode;
  cusparseGetPointerMode(handle /*cusparseHandle_t*/, &mode);
  // End
}
