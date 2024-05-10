#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m,
          int64_t n, cudaDataType a_type, const void *a, int64_t lda,
          cudaDataType compute_type) {
  // Start
  size_t device_buffer_size;
  size_t host_buffer_size;
  cusolverDnXgetrf_bufferSize(
      handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
      m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/, a /*const void **/,
      lda /*int64_t*/, compute_type /*cudaDataType*/,
      &device_buffer_size /*size_t **/, &host_buffer_size /*size_t **/);
  // End
}
