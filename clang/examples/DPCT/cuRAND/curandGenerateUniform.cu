#include "curand.h"

void test(float *pf, size_t s) {
  // Start
  curandGenerator_t g;
  curandGenerateUniform(g /*curandGenerator_t*/, pf /*float **/, s /*size_t*/);
  // End
}
