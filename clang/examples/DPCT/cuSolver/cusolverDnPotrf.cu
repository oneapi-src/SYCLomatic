#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params,
          cublasFillMode_t uplo, int64_t n, cudaDataType a_type, void *a,
          int64_t lda, cudaDataType compute_type, void *buffer,
          size_t buffer_size, int *info) {
  // Start
  cusolverDnPotrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
                  uplo /*cublasFillMode_t*/, n /*int64_t*/,
                  a_type /*cudaDataType*/, a /*void **/, lda /*int64_t*/,
                  compute_type /*cudaDataType*/, buffer /*void **/,
                  buffer_size /*size_t*/, info /*int **/);
  // End
}
