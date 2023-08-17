#include "cublas_v2.h"

void test(cublasHandle_t handle, float *a, float *b, float *c, float *s) {
  // Start
  cublasSrotg(handle /*cublasHandle_t*/, a /*float **/, b /*float **/,
              c /*float **/, s /*float **/);
  // End
}
