#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasMath_t precision) {
  // Start
  cublasSetMathMode(handle /*cublasHandle_t*/, precision /*cublasMath_t*/);
  // End
}
