#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasPointerMode_t *host_device) {
  // Start
  cublasGetPointerMode(handle /*cublasHandle_t*/,
                       host_device /*cublasPointerMode_t **/);
  // End
}
