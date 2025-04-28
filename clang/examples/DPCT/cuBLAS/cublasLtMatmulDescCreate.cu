#include "cublasLt.h"

void test(cublasLtMatmulDesc_t *mm_esc, cublasComputeType_t compute_type,
          cudaDataType_t scale_type) {
  // Start
  cublasLtMatmulDescCreate(mm_esc /*cublasLtMatmulDesc_t **/,
                           compute_type /*cublasComputeType_t*/,
                           scale_type /*cudaDataType_t*/);
  // End
}
