#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
          int n, int k, const float *alpha, const void *a, cudaDataType a_type,
          int lda, const float *beta, void *c, cudaDataType c_type, int ldc) {
  // Start
  cublasCherk3mEx(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
                  trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
                  alpha /*const float **/, a /*const void **/,
                  a_type /*cudaDataType*/, lda /*int*/, beta /*const float **/,
                  c /*void **/, c_type /*cudaDataType*/, ldc /*int*/);
  // End
}
