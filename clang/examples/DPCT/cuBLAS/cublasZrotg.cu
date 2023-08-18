#include "cublas_v2.h"

void test(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b,
          double *c, cuDoubleComplex *s) {
  // Start
  cublasZrotg(handle /*cublasHandle_t*/, a /*cuDoubleComplex **/,
              b /*cuDoubleComplex **/, c /*double **/, s /*cuDoubleComplex **/);
  // End
}
