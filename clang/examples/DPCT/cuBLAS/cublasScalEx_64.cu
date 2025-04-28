#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const void *alpha,
          cudaDataType alphatype, void *x, cudaDataType xtype, int64_t incx,
          cudaDataType computetype) {
  // Start
  cublasScalEx_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                  alpha /*const void **/, alphatype /*cudaDataType*/,
                  x /*void **/, xtype /*cudaDataType*/, incx /*int64_t*/,
                  computetype /*cudaDataType*/);
  // End
}
