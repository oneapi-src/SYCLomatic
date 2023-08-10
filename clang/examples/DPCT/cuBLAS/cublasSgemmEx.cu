#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const float *alpha,
          const void *a, cudaDataType atype, int lda, const void *b,
          cudaDataType btype, int ldb, const float *beta, void *c,
          cudaDataType ctype, int ldc) {
  // Start
  cublasSgemmEx(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
                alpha /*const float **/, a /*const void **/,
                atype /*cudaDataType*/, lda /*int*/, b /*const void **/,
                btype /*cudaDataType*/, ldb /*int*/, beta /*const float **/,
                c /*void **/, ctype /*cudaDataType*/, ldc /*int*/);
  // End
}
