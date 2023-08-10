#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const void *alpha,
          cudaDataType alphatype, const void *x, cudaDataType xtype, int incx,
          void *y, cudaDataType ytype, int incy, cudaDataType computetype) {
  // Start
  cublasAxpyEx(handle /*cublasHandle_t*/, n /*int*/, alpha /*const void **/,
               alphatype /*cudaDataType*/, x /*const void **/,
               xtype /*cudaDataType*/, incx /*int*/, y /*void **/,
               ytype /*cudaDataType*/, incy /*int*/,
               computetype /*cudaDataType*/);
  // End
}
