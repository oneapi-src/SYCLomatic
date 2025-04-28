#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const void *alpha,
          cudaDataType alphatype, const void *x, cudaDataType xtype,
          int64_t incx, void *y, cudaDataType ytype, int64_t incy,
          cudaDataType computetype) {
  // Start
  cublasAxpyEx_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                  alpha /*const void **/, alphatype /*cudaDataType*/,
                  x /*const void **/, xtype /*cudaDataType*/, incx /*int64_t*/,
                  y /*void **/, ytype /*cudaDataType*/, incy /*int64_t*/,
                  computetype /*cudaDataType*/);
  // End
}
