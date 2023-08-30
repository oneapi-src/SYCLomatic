#include "cublas_v2.h"

void test(cublasHandle_t handle, int *ver) {
  // Start
  cublasGetVersion(handle /*cublasHandle_t*/, ver /*int **/);
  // End
}
