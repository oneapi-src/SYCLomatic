#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx,
          double *res) {
  // Start
  cublasDzasum(handle /*cublasHandle_t*/, n /*int*/,
               x /*const cuDoubleComplex **/, incx /*int*/, res /*double **/);
  // End
}
