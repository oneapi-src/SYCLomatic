#include "cublasLt.h"

void test(cublasLtMatrixTransformDesc_t trans_desc) {
  // Start
  cublasLtMatrixTransformDescDestroy(
      trans_desc /*cublasLtMatrixTransformDesc_t*/);
  // End
}
