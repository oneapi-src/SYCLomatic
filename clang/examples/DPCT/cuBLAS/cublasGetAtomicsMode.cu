// Not migrate: The API is Removed because this it is redundant in SYCL.
#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasAtomicsMode_t *atomics) {
  // Start
  cublasGetAtomicsMode(handle /*cublasHandle_t*/,
                       atomics /*cublasAtomicsMode_t **/);
  // End
}
