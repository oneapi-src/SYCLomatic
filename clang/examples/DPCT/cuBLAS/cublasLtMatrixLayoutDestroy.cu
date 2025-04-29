#include "cublasLt.h"

void test(cublasLtMatrixLayout_t layout) {
  // Start
  cublasLtMatrixLayoutDestroy(layout /*cublasLtMatrixLayout_t*/);
  // End
}
