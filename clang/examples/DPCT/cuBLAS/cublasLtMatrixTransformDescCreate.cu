#include "cublasLt.h"

void test(cublasLtMatrixTransformDesc_t *trans_desc, cudaDataType scale_type) {
  // Start
  cublasLtMatrixTransformDescCreate(
      trans_desc /*cublasLtMatrixTransformDesc_t **/,
      scale_type /*cudaDataType*/);
  // End
}
