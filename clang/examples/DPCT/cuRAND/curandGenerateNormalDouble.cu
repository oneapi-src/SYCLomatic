#include "curand.h"

void test(double *pd, size_t s, double d1, double d2) {
  // Start
  curandGenerator_t g;
  curandGenerateNormalDouble(g /*curandGenerator_t*/, pd /*double **/,
                             s /*size_t*/, d1 /*double*/, d2 /*double*/);
  // End
}
