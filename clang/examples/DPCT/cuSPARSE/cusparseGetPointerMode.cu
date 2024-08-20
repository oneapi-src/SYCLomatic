#include "cusparse.h"

void test(cusparseHandle_t handle, cusparsePointerMode_t *mode) {
  // Start
  cusparseGetPointerMode(handle /*cusparseHandle_t*/,
                         mode /*cusparsePointerMode_t **/);
  // End
}
