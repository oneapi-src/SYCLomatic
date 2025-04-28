#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, void *x, cudaDataType xtype,
          int64_t incx, void *y, cudaDataType ytype, int64_t incy,
          const void *c, const void *s, cudaDataType cstype,
          cudaDataType computetype) {
  // Start
  cublasRotEx_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*void **/,
                 xtype /*cudaDataType*/, incx /*int64_t*/, y /*void **/,
                 ytype /*cudaDataType*/, incy /*int64_t*/, c /*const void **/,
                 s /*const void **/, cstype /*cudaDataType*/,
                 computetype /*cudaDataType*/);
  // End
}
