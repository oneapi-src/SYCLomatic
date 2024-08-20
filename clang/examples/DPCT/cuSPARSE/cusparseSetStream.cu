#include "cusparse.h"

void test(cusparseHandle_t handle, cudaStream_t s) {
  // Start
  cusparseSetStream(handle /*cusparseHandle_t*/, s /*cudaStream_t*/);
  // End
}
