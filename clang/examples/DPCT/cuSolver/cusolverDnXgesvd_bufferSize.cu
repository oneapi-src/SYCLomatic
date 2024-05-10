#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params,
          signed char jobu, signed char jobvt, int64_t m, int64_t n,
          cudaDataType a_type, const void *a, int64_t lda, cudaDataType s_type,
          const void *s, cudaDataType u_type, const void *u, int64_t ldu,
          cudaDataType vt_type, const void *vt, int64_t ldvt,
          cudaDataType compute_type) {
  // Start
  size_t device_buffer_size;
  size_t host_buffer_size;
  cusolverDnXgesvd_bufferSize(
      handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
      jobu /*signed char*/, jobvt /*signed char*/, m /*int64_t*/, n /*int64_t*/,
      a_type /*cudaDataType*/, a /*const void **/, lda /*int64_t*/,
      s_type /*cudaDataType*/, s /*const void **/, u_type /*cudaDataType*/,
      u /*const void **/, ldu /*int64_t*/, vt_type /*cudaDataType*/,
      vt /*const void **/, ldvt /*int64_t*/, compute_type /*cudaDataType*/,
      &device_buffer_size /*size_t **/, &host_buffer_size /*size_t **/);
  // End
}
