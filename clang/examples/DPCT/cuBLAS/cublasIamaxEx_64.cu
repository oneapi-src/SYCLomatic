#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const void *x, cudaDataType x_type,
          int64_t incx, int64_t *result) {
  // Start
  cublasIamaxEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
                   x_type /*cudaDataType*/, incx /*int64_t*/,
                   result /*int64_t **/);
  // End
}
