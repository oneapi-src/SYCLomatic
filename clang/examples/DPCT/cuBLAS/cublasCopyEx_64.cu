#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const void *x, cudaDataType x_type,
          int64_t incx, void *y, cudaDataType y_type, int64_t incy) {
  // Start
  cublasCopyEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
                  x_type /*cudaDataType*/, incx /*int64_t*/, y /*void **/,
                  y_type /*cudaDataType*/, incy /*int64_t*/);
  // End
}
