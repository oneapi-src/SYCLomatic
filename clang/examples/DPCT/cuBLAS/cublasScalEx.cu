#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const void *alpha,
          cudaDataType alphatype, void *x, cudaDataType xtype, int incx,
          cudaDataType computetype) {
  // Start
  cublasScalEx(handle /*cublasHandle_t*/, n /*int*/, alpha /*const void **/,
               alphatype /*cudaDataType*/, x /*void **/, xtype /*cudaDataType*/,
               incx /*int*/, computetype /*cudaDataType*/);
  // End
}
