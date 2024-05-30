#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params,
          cublasFillMode_t uplo, int64_t n, cudaDataType a_type, const void *a,
          int64_t lda, cudaDataType compute_type) {
  // Start
  size_t buffer_size;
  cusolverDnPotrf_bufferSize(
      handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
      uplo /*cublasFillMode_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
      a /*const void **/, lda /*int64_t*/, compute_type /*cudaDataType*/,
      &buffer_size /*size_t **/);
  // End
}
