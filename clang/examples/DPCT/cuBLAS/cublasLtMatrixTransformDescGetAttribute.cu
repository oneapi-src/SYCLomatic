#include "cublasLt.h"

void test(cublasLtMatrixTransformDesc_t transformDesc,
          cublasLtMatrixTransformDescAttributes_t attr, void *buf,
          size_t sizeInBytes, size_t *sizeWritten) {
  // Start
  cublasLtMatrixTransformDescGetAttribute(
      transformDesc /*cublasLtMatrixTransformDesc_t*/,
      attr /*cublasLtMatrixTransformDescAttributes_t*/, buf /*void **/,
      sizeInBytes /*size_t*/, sizeWritten /*size_t **/);
  // End
}
