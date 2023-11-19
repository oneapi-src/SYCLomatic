#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasPointerMode_t host_device) {
  // Start
  cublasSetPointerMode(handle /*cublasHandle_t*/,
                       host_device /*cublasPointerMode_t*/);
  // End
}
