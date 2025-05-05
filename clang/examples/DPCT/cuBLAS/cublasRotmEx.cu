#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, void *x, cudaDataType x_type, int incx,
          void *y, cudaDataType y_type, int incy, const void *param,
          cudaDataType param_type, cudaDataType execution_type) {
  // Start
  cublasRotmEx(handle /*cublasHandle_t*/, n /*int*/, x /*void **/,
               x_type /*cudaDataType*/, incx /*int*/, y /*void **/,
               y_type /*cudaDataType*/, incy /*int*/, param /*const void **/,
               param_type /*cudaDataType*/, execution_type /*cudaDataType*/);
  // End
}
