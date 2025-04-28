#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
          int64_t n, int64_t k, const cuComplex *alpha, const void *a,
          cudaDataType a_type, int64_t lda, const cuComplex *beta, void *c,
          cudaDataType c_type, int64_t ldc) {
  // Start
  cublasCsyrkEx_64(handle /*cublasHandle_t*/, uplo /*cublasFillMode_t*/,
                   trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
                   alpha /*const cuComplex **/, a /*const void **/,
                   a_type /*cudaDataType*/, lda /*int64_t*/,
                   beta /*const cuComplex **/, c /*void **/,
                   c_type /*cudaDataType*/, ldc /*int64_t*/);
  // End
}
