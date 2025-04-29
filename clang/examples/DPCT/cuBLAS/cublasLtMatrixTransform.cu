#include "cublasLt.h"

void test(cublasLtHandle_t lthandle, cublasLtMatrixTransformDesc_t trans_desc,
          const void *alpha, const void *a, cublasLtMatrixLayout_t a_desc,
          const void *beta, const void *b, cublasLtMatrixLayout_t b_desc,
          void *c, cublasLtMatrixLayout_t c_desc, cudaStream_t stream) {
  // Start
  cublasLtMatrixTransform(
      lthandle /*cublasLtHandle_t*/,
      trans_desc /*cublasLtMatrixTransformDesc_t*/, alpha /*const void **/,
      a /*const void **/, a_desc /*cublasLtMatrixLayout_t*/,
      beta /*const void **/, b /*const void **/,
      b_desc /*cublasLtMatrixLayout_t*/, c /*void **/,
      c_desc /*cublasLtMatrixLayout_t*/, stream /*cudaStream_t*/);
  // End
}
