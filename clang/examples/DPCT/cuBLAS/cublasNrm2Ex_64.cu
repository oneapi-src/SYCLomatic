#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const void *x, cudaDataType xtype,
          int64_t incx, void *res, cudaDataType restype,
          cudaDataType computetype) {
  // Start
  cublasNrm2Ex_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const void **/,
                  xtype /*cudaDataType*/, incx /*int64_t*/, res /*void **/,
                  restype /*cudaDataType*/, computetype /*cudaDataType*/);
  // End
}
