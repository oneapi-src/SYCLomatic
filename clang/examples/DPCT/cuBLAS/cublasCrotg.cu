#include "cublas_v2.h"

void test(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c,
          cuComplex *s) {
  // Start
  cublasCrotg(handle /*cublasHandle_t*/, a /*cuComplex **/, b /*cuComplex **/,
              c /*float **/, s /*cuComplex **/);
  // End
}
