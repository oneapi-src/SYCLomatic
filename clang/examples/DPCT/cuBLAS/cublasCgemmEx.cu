#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t transa,
          cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha,
          const void *a, cudaDataType atype, int lda, const void *b,
          cudaDataType btype, int ldb, const cuComplex *beta, void *c,
          cudaDataType ctype, int ldc) {
  // Start
  cublasCgemmEx(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
                transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
                alpha /*const cuComplex **/, a /*const void **/,
                atype /*cudaDataType*/, lda /*int*/, b /*const void **/,
                btype /*cudaDataType*/, ldb /*int*/, beta /*const cuComplex **/,
                c /*void **/, ctype /*cudaDataType*/, ldc /*int*/);
  // End
}
