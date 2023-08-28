#include "curand.h"

void test(float *pf, size_t s, float f1, float f2) {
  // Start
  curandGenerator_t g;
  curandGenerateNormal(g /*curandGenerator_t*/, pf /*float **/, s /*size_t*/,
                       f1 /*float*/, f2 /*float*/);
  // End
}
