#include "cublasLt.h"

void test(cublasLtMatrixTransformDesc_t transformDesc,
          cublasLtMatrixTransformDescAttributes_t attr, void *buf,
          size_t sizeInBytes) {
  // Start
  cublasLtMatrixTransformDescSetAttribute(
      transformDesc /*cublasLtMatrixTransformDesc_t*/,
      attr /*cublasLtMatrixTransformDescAttributes_t*/, buf /*void **/,
      sizeInBytes /*size_t*/);
  // End
}
