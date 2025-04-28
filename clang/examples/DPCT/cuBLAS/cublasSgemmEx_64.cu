#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
          const float *alpha, const void *a, cudaDataType atype, int64_t lda,
          const void *b, cudaDataType btype, int64_t ldb, const float *beta,
          void *c, cudaDataType ctype, int64_t ldc) {
  // Start
  cublasSgemmEx_64(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                   transb /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
                   k /*int64_t*/, alpha /*const float **/, a /*const void **/,
                   atype /*cudaDataType*/, lda /*int64_t*/, b /*const void **/,
                   btype /*cudaDataType*/, ldb /*int64_t*/,
                   beta /*const float **/, c /*void **/, ctype /*cudaDataType*/,
                   ldc /*int64_t*/);
  // End
}
