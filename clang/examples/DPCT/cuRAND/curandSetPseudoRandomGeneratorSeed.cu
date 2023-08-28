#include "curand.h"

void test(unsigned long long ull) {
  // Start
  curandGenerator_t g;
  curandSetPseudoRandomGeneratorSeed(g /*curandGenerator_t*/,
                                     ull /*unsigned long long*/);
  // End
}
