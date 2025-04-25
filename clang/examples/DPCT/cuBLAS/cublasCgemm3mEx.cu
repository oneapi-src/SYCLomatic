#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans_a,
          cublasOperation_t trans_b, int m, int n, int k,
          const cuComplex *alpha, const void *a, cudaDataType a_type, int lda,
          const void *b, cudaDataType b_type, int ldb, const cuComplex *beta,
          void *c, cudaDataType c_type, int ldc) {
  // Start
  cublasCgemm3mEx(handle /*cublasHandle_t*/, trans_a /*cublasOperation_t*/,
                  trans_b /*cublasOperation_t*/, m /*int*/, n /*int*/,
                  k /*int*/, alpha /*const cuComplex **/, a /*const void **/,
                  a_type /*cudaDataType*/, lda /*int*/, b /*const void **/,
                  b_type /*cudaDataType*/, ldb /*int*/,
                  beta /*const cuComplex **/, c /*void **/,
                  c_type /*cudaDataType*/, ldc /*int*/);
  // End
}
