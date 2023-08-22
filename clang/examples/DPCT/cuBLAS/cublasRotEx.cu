#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, void *x, cudaDataType xtype, int incx,
          void *y, cudaDataType ytype, int incy, const void *c, const void *s,
          cudaDataType cstype, cudaDataType computetype) {
  // Start
  cublasRotEx(handle /*cublasHandle_t*/, n /*int*/, x /*void **/,
              xtype /*cudaDataType*/, incx /*int*/, y /*void **/,
              ytype /*cudaDataType*/, incy /*int*/, c /*const void **/,
              s /*const void **/, cstype /*cudaDataType*/,
              computetype /*cudaDataType*/);
  // End
}
