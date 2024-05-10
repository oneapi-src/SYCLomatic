#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params,
          cublasOperation_t trans, int64_t n, int64_t nrhs, cudaDataType a_type,
          const void *a, int64_t lda, const int64_t *ipiv, cudaDataType b_type,
          void *b, int64_t ldb, int *info) {
  // Start
  cusolverDnXgetrs(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
                   trans /*cublasOperation_t*/, n /*int64_t*/, nrhs /*int64_t*/,
                   a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
                   ipiv /*const int64_t **/, b_type /*cudaDataType*/,
                   b /*void **/, ldb /*int64_t*/, info /*int **/);
  // End
}
