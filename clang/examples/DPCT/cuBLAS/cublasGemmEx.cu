#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const void *alpha,
          const void *a, cudaDataType atype, int lda, const void *b,
          cudaDataType btype, int ldb, const void *beta, void *c,
          cudaDataType ctype, int ldc,
          cublasComputeType_t computetype_computeType_t,
          cudaDataType computetype_dataType, cublasGemmAlgo_t algo) {
  // Start
  cublasGemmEx(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
               alpha /*const void **/, a /*const void **/,
               atype /*cudaDataType*/, lda /*int*/, b /*const void **/,
               btype /*cudaDataType*/, ldb /*int*/, beta /*const void **/,
               c /*void **/, ctype /*cudaDataType*/, ldc /*int*/,
               computetype_computeType_t /*cublasComputeType_t*/,
               algo /*cublasGemmAlgo_t*/);
  cublasGemmEx(
      handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
      transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
      alpha /*const void **/, a /*const void **/, atype /*cudaDataType*/,
      lda /*int*/, b /*const void **/, btype /*cudaDataType*/, ldb /*int*/,
      beta /*const void **/, c /*void **/, ctype /*cudaDataType*/, ldc /*int*/,
      computetype_dataType /*cudaDataType*/, algo /*cublasGemmAlgo_t*/);
  // End
}
