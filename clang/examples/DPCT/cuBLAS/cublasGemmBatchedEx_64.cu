#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
          const void *alpha, const void *const *a, cudaDataType atype,
          int64_t lda, const void *const *b, cudaDataType btype, int64_t ldb,
          const void *beta, void *const *c, cudaDataType ctype, int64_t ldc,
          int64_t group_count, cublasComputeType_t computetype_computeType_t,
          cudaDataType computetype_dataType, cublasGemmAlgo_t algo) {
  // Start
  cublasGemmBatchedEx_64(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/, k /*int64_t*/,
      alpha /*const void **/, a /*const void *const **/, atype /*cudaDataType*/,
      lda /*int64_t*/, b /*const void *const **/, btype /*cudaDataType*/,
      ldb /*int64_t*/, beta /*const void **/, c /*void *const **/,
      ctype /*cudaDataType*/, ldc /*int64_t*/, group_count /*int64_t*/,
      computetype_computeType_t /*cublasComputeType_t*/,
      algo /*cublasGemmAlgo_t*/);
  // End
}
