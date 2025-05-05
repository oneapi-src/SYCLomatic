#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const void *x, cudaDataType x_type,
          int incx, int *result) {
  // Start
  cublasIaminEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
                x_type /*cudaDataType*/, incx /*int*/, result /*int **/);
  // End
}
