#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m,
          int64_t n, cudaDataType a_type, void *a, int64_t lda,
          cudaDataType tau_type, void *tau, cudaDataType compute_type,
          void *device_buffer, size_t device_buffer_size, void *host_buffer,
          size_t host_buffer_size, int *info) {
  // Start
  cusolverDnXgeqrf(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
                   m /*int64_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
                   a /*void **/, lda /*int64_t*/, tau_type /*cudaDataType*/,
                   tau /*void **/, compute_type /*cudaDataType*/,
                   device_buffer /*void **/, device_buffer_size /*size_t*/,
                   host_buffer /*void **/, host_buffer_size /*size_t*/,
                   info /*int **/);
  // End
}
