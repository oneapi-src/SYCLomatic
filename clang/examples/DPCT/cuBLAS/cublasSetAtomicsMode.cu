#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasAtomicsMode_t atomics) {
  // Start
  cublasSetAtomicsMode(handle /*cublasHandle_t*/,
                       atomics /*cublasAtomicsMode_t*/);
  // End
}
