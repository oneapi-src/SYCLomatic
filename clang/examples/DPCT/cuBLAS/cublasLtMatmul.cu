#include "cublasLt.h"

void test(cublasLtHandle_t lthandle, cublasLtMatmulDesc_t mm_desc,
          const void *alpha, const void *a, cublasLtMatrixLayout_t a_desc,
          const void *b, cublasLtMatrixLayout_t b_desc, const void *beta,
          const void *c, cublasLtMatrixLayout_t c_desc, void *d,
          cublasLtMatrixLayout_t d_desc, const cublasLtMatmulAlgo_t *algo,
          void *workspace, size_t workspace_size, cudaStream_t stream) {
  // Start
  cublasLtMatmul(
      lthandle /*cublasLtHandle_t*/, mm_desc /*cublasLtMatmulDesc_t*/,
      alpha /*const void **/, a /*const void **/,
      a_desc /*cublasLtMatrixLayout_t*/, b /*const void **/,
      b_desc /*cublasLtMatrixLayout_t*/, beta /*const void **/,
      c /*const void **/, c_desc /*cublasLtMatrixLayout_t*/, d /*void **/,
      d_desc /*cublasLtMatrixLayout_t*/, algo /*const cublasLtMatmulAlgo_t **/,
      workspace /*void **/, workspace_size /*size_t*/, stream /*cudaStream_t*/);
  // End
}
