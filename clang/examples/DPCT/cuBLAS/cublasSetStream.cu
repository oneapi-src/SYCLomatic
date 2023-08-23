#include "cublas_v2.h"

void test(cublasHandle_t handle, cudaStream_t stream) {
  // Start
  cublasSetStream(handle /*cublasHandle_t*/, stream /*cudaStream_t*/);
  // End
}
