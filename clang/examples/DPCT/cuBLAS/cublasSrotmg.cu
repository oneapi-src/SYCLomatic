#include "cublas_v2.h"

void test(cublasHandle_t handle, float *d1, float *d2, float *x1,
          const float *y1, float *param) {
  // Start
  cublasSrotmg(handle /*cublasHandle_t*/, d1 /*float **/, d2 /*float **/,
               x1 /*float **/, y1 /*const float **/, param /*float **/);
  // End
}
