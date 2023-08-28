#include "curand.h"

void test(unsigned int *pu, size_t s, double d) {
  // Start
  curandGenerator_t g;
  curandGeneratePoisson(g /*curandGenerator_t*/, pu /*unsigned int **/,
                        s /*size_t*/, d /*double*/);
  // End
}
