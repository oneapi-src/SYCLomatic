#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params,
          cusolverEigMode_t jobz, cusolverEigRange_t range,
          cublasFillMode_t uplo, int n, cudaDataType a_type, void *a,
          int64_t lda, void *vl, void *vu, int64_t il, int64_t iu,
          int64_t *h_meig, cudaDataType w_type, void *w,
          cudaDataType compute_type, void *buffer, size_t buffer_size,
          int *info) {
  // Start
  cusolverDnSyevdx(handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
                   jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
                   uplo /*cublasFillMode_t*/, n /*int64_t*/,
                   a_type /*cudaDataType*/, a /*void **/, lda /*int64_t*/,
                   vl /*void **/, vu /*void **/, il /*int64_t*/, iu /*int64_t*/,
                   h_meig /*int64_t **/, w_type /*cudaDataType*/, w /*void **/,
                   compute_type /*cudaDataType*/, buffer /*void **/,
                   buffer_size /*size_t*/, info /*int **/);
  // End
}
