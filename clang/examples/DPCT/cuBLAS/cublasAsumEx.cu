#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const void *x, cudaDataType x_type,
          int incx, void *result, cudaDataType result_type,
          cudaDataType execution_type) {
  // Start
  cublasAsumEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
               x_type /*cudaDataType*/, incx /*int*/, result /*void **/,
               result_type /*cudaDataType*/, execution_type /*cudaDataType*/);
  // End
}
