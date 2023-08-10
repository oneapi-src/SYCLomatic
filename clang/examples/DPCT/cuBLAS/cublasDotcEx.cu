#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const void *x, cudaDataType xtype,
          int incx, const void *y, cudaDataType ytype, int incy, void *res,
          cudaDataType restype, cudaDataType computetype) {
  // Start
  cublasDotcEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
               xtype /*cudaDataType*/, incx /*int*/, y /*const void **/,
               ytype /*cudaDataType*/, incy /*int*/, res /*void **/,
               restype /*cudaDataType*/, computetype /*cudaDataType*/);
  // End
}
