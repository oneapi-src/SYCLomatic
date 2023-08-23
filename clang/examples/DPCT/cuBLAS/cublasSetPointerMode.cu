// Not migrate: The API is Removed because this it is redundant in SYCL.
#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasPointerMode_t host_device) {
  // Start
  cublasSetPointerMode(handle /*cublasHandle_t*/,
                       host_device /*cublasPointerMode_t*/);
  // End
}
