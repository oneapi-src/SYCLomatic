#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params,
          cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
          cudaDataType a_type, const void *a, int64_t lda, cudaDataType w_type,
          const void *w, cudaDataType compute_type) {
  // Start
  size_t device_buffer_size;
  size_t host_buffer_size;
  cusolverDnXsyevd_bufferSize(
      handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
      jobz /*cusolverEigMode_t*/, uplo /*cublasFillMode_t*/, n /*int64_t*/,
      a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
      w_type /*cudaDataType*/, w /*const void **/,
      compute_type /*cudaDataType*/, &device_buffer_size /*size_t **/,
      &host_buffer_size /*size_t **/);
  // End
}
