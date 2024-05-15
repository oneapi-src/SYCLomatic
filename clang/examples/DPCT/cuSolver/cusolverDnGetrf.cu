#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m,
          int64_t n, cudaDataType a_type, void *a, int64_t lda, int64_t *ipiv,
          cudaDataType compute_type, void *buffer, size_t buffer_size,
          int *info) {
  // Start
  cusolverDnGetrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
                  m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
                  a /*void **/, lda /*int64_t*/, ipiv /*int64_t **/,
                  compute_type /*cudaDataType*/, buffer /*void **/,
                  buffer_size /*size_t*/, info /*int **/);
  // End
}
