#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
          const void *alpha, const void *a, cudaDataType atype, int64_t lda,
          long long int stridea, const void *b, cudaDataType btype, int64_t ldb,
          long long int strideb, const void *beta, void *c, cudaDataType ctype,
          int64_t ldc, long long int stridec, int64_t group_count,
          cublasComputeType_t computetype_computeType_t,
          cudaDataType computetype_dataType, cublasGemmAlgo_t algo) {
  // Start
  cublasGemmStridedBatchedEx_64(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/, k /*int64_t*/,
      alpha /*const void **/, a /*const void **/, atype /*cudaDataType*/,
      lda /*int64_t*/, stridea /*long long int*/, b /*const void **/,
      btype /*cudaDataType*/, ldb /*int64_t*/, strideb /*long long int*/,
      beta /*const void **/, c /*void **/, ctype /*cudaDataType*/,
      ldc /*int64_t*/, stridec /*long long int*/, group_count /*int64_t*/,
      computetype_computeType_t /*cublasComputeType_t*/,
      algo /*cublasGemmAlgo_t*/);
  // End
}
