#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans_a,
          cublasOperation_t trans_b, int64_t m, int64_t n, int64_t k,
          const cuComplex *alpha, const void *a, cudaDataType a_type,
          int64_t lda, const void *b, cudaDataType b_type, int64_t ldb,
          const cuComplex *beta, void *c, cudaDataType c_type, int64_t ldc) {
  // Start
  cublasCgemm3mEx_64(
      handle /*cublasHandle_t*/, trans_a /*cublasOperation_t*/,
      trans_b /*cublasOperation_t*/, m /*int64_t*/, n /*int64_t*/,
      k /*int64_t*/, alpha /*const cuComplex **/, a /*const void **/,
      a_type /*cudaDataType*/, lda /*int64_t*/, b /*const void **/,
      b_type /*cudaDataType*/, ldb /*int64_t*/, beta /*const cuComplex **/,
      c /*void **/, c_type /*cudaDataType*/, ldc /*int64_t*/);
  // End
}
