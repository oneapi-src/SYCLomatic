#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const void *alpha,
          const void *const *a, cudaDataType atype, int lda,
          const void *const *b, cudaDataType btype, int ldb, const void *beta,
          void *const *c, cudaDataType ctype, int ldc, int group_count,
          cublasComputeType_t computetype_computeType_t,
          cudaDataType computetype_dataType, cublasGemmAlgo_t algo) {
  // Start
  cublasGemmBatchedEx(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const void **/, a /*const void *const **/, atype /*cudaDataType*/,
      lda /*int*/, b /*const void *const **/, btype /*cudaDataType*/,
      ldb /*int*/, beta /*const void **/, c /*void *const **/,
      ctype /*cudaDataType*/, ldc /*int*/, group_count /*int*/,
      computetype_computeType_t /*cublasComputeType_t*/,
      algo /*cublasGemmAlgo_t*/);
  cublasGemmBatchedEx(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const void **/, a /*const void *const **/, atype /*cudaDataType*/,
      lda /*int*/, b /*const void *const **/, btype /*cudaDataType*/,
      ldb /*int*/, beta /*const void **/, c /*void *const **/,
      ctype /*cudaDataType*/, ldc /*int*/, group_count /*int*/,
      computetype_dataType /*cudaDataType*/, algo /*cublasGemmAlgo_t*/);
  // End
}
