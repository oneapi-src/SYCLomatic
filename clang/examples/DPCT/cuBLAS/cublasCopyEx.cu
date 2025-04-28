#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const void *x, cudaDataType x_type,
          int incx, void *y, cudaDataType y_type, int incy) {
  // Start
  cublasCopyEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
               x_type /*cudaDataType*/, incx /*int*/, y /*void **/,
               y_type /*cudaDataType*/, incy /*int*/);
  // End
}
