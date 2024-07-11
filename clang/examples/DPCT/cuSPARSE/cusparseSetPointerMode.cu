#include "cusparse.h"

void test(cusparseHandle_t handle, cusparsePointerMode_t mode) {
  // Start
  cusparseSetPointerMode(handle /*cusparseHandle_t*/,
                         mode /*cusparsePointerMode_t*/);
  // End
}
