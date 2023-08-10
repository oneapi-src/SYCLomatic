#include "cublas_v2.h"

void test(cublasHandle_t handle, double *d1, double *d2, double *x1,
          const double *y1, double *param) {
  // Start
  cublasDrotmg(handle /*cublasHandle_t*/, d1 /*double **/, d2 /*double **/,
               x1 /*double **/, y1 /*const double **/, param /*double **/);
  // End
}
