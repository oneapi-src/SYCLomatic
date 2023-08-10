#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const float *alpha,
          const void *a, cudaDataType_t atype, int lda, const void *b,
          cudaDataType_t btype, int ldb, const float *beta, void *c,
          cudaDataType_t ctype, int ldc) {
  // Start
  cublasSgemmEx(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
                alpha /*const float **/, a /*const void **/,
                atype /*cudaDataType_t*/, lda /*int*/, b /*const void **/,
                btype /*cudaDataType_t*/, ldb /*int*/, beta /*const float **/,
                c /*const void **/, ctype /*cudaDataType_t*/, ldc /*int*/);
  // End
}
