#include "curand.h"

void test(curandOrdering_t o) {
  // Start
  curandGenerator_t g;
  curandSetGeneratorOrdering(g /*curandGenerator_t*/, o /*curandOrdering_t*/);
  // End
}
