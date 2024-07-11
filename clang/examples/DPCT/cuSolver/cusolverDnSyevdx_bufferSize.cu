#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, cusolverDnParams_t params,
          cusolverEigMode_t jobz, cusolverEigRange_t range,
          cublasFillMode_t uplo, int n, cudaDataType a_type, const void *a,
          int64_t lda, void *vl, void *vu, int64_t il, int64_t iu,
          int64_t *h_meig, cudaDataType w_type, const void *w,
          cudaDataType compute_type) {
  // Start
  size_t buffer_size;
  cusolverDnSyevdx_bufferSize(
      handle /*cusolverDnHandle_t*/, params /*cusolverDnParams_t*/,
      jobz /*cusolverEigMode_t*/, range /*cusolverEigRange_t*/,
      uplo /*cublasFillMode_t*/, n /*int64_t*/, a_type /*cudaDataType*/,
      a /*const void **/, lda /*int64_t*/, vl /*void **/, vu /*void **/,
      il /*int64_t*/, iu /*int64_t*/, h_meig /*int64_t **/,
      w_type /*cudaDataType*/, w /*const void **/,
      compute_type /*cudaDataType*/, &buffer_size /*size_t **/);
  // End
}
