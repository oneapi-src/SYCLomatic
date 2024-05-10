#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params,
          signed char jobu, signed char jobvt, int64_t m, int64_t n,
          cudaDataType a_type, void *a, int64_t lda, cudaDataType s_type,
          void *s, cudaDataType u_type, void *u, int64_t ldu,
          cudaDataType vt_type, void *vt, int64_t ldvt,
          cudaDataType compute_type, void *device_buffer,
          size_t device_buffer_size, void *host_buffer, size_t host_buffer_size,
          int *info) {
  // Start
  cusolverDnXgesvd(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
                   jobu /*signed char*/, jobvt /*signed char*/, m /*int64_t*/,
                   n /*int64_t*/, a_type /*cudaDataType*/, a /*void **/,
                   lda /*int64_t*/, s_type /*cudaDataType*/, s /*void **/,
                   u_type /*cudaDataType*/, u /*void **/, ldu /*int64_t*/,
                   vt_type /*cudaDataType*/, vt /*void **/, ldvt /*int64_t*/,
                   compute_type /*cudaDataType*/, device_buffer /*void **/,
                   device_buffer_size /*size_t*/, host_buffer /*void **/,
                   host_buffer_size /*size_t*/, info /*int **/);
  // End
}
