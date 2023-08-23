#include "cublas_v2.h"

void test(cublasHandle_t handle, double *a, double *b, double *c, double *s) {
  // Start
  cublasDrotg(handle /*cublasHandle_t*/, a /*double **/, b /*double **/,
              c /*double **/, s /*double **/);
  // End
}
