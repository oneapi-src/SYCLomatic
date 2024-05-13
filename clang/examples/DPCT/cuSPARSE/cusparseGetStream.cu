#include "cusparse.h"

void test(cusparseHandle_t handle) {
  // Start
  cudaStream_t s;
  cusparseGetStream(handle /*cusparseHandle_t*/, &s /*cudaStream_t **/);
  // End
}
