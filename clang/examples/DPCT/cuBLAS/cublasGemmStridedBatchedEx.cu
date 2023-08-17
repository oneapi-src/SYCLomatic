#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const void *alpha,
          const void *a, cudaDataType atype, int lda, long long int stridea,
          const void *b, cudaDataType btype, int ldb, long long int strideb,
          const void *beta, void *c, cudaDataType ctype, int ldc,
          long long int stridec, int group_count,
          cublasComputeType_t computetype_computeType_t,
          cudaDataType computetype_dataType, cublasGemmAlgo_t algo) {
  // Start
  cublasGemmStridedBatchedEx(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const void **/, a /*const void **/, atype /*cudaDataType*/,
      lda /*int*/, stridea /*long long int*/, b /*const void **/,
      btype /*cudaDataType*/, ldb /*int*/, strideb /*long long int*/,
      beta /*const void **/, c /*void **/, ctype /*cudaDataType*/, ldc /*int*/,
      stridec /*long long int*/, group_count /*int*/,
      computetype_computeType_t /*cublasComputeType_t*/,
      algo /*cublasGemmAlgo_t*/);
  cublasGemmStridedBatchedEx(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const void **/, a /*const void **/, atype /*cudaDataType*/,
      lda /*int*/, stridea /*long long int*/, b /*const void **/,
      btype /*cudaDataType*/, ldb /*int*/, strideb /*long long int*/,
      beta /*const void **/, c /*void **/, ctype /*cudaDataType*/, ldc /*int*/,
      stridec /*long long int*/, group_count /*int*/,
      computetype_dataType /*cudaDataType*/, algo /*cublasGemmAlgo_t*/);
  // End
}
