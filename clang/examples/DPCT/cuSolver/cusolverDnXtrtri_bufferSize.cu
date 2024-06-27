#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cublasFillMode_t uplo,
          cublasDiagType_t diag, int64_t n, cudaDataType a_type, void *a,
          int64_t lda) {
  // Start
  size_t device_buffer_size;
  size_t host_buffer_size;
  cusolverDnXtrtri_bufferSize(
      handle /*cusolverDnHandle_t*/, uplo /*cublasFillMode_t*/,
      diag /*cublasDiagType_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
      a /*void **/, lda /*int64_t*/, &device_buffer_size /*size_t **/,
      &host_buffer_size /*size_t **/);
  // End
}
