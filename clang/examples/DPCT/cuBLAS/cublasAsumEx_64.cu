#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const void *x, cudaDataType x_type,
          int64_t incx, void *result, cudaDataType result_type,
          cudaDataType execution_type) {
  // Start
  cublasAsumEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
                  x_type /*cudaDataType*/, incx /*int64_t*/, result /*void **/,
                  result_type /*cudaDataType*/,
                  execution_type /*cudaDataType*/);
  // End
}
