#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const void *x, cudaDataType xtype,
          int incx, void *res, cudaDataType restype, cudaDataType computetype) {
  // Start
  cublasNrm2Ex(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
               xtype /*cudaDataType*/, incx /*int*/, res /*void **/,
               restype /*cudaDataType*/, computetype /*cudaDataType*/);
  // End
}
