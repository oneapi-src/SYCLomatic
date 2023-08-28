#include "curand.h"

void test(unsigned long long *ull, size_t s) {
  // Start
  curandGenerator_t g;
  curandGenerateLongLong(g /*curandGenerator_t*/, ull /*unsigned long long **/,
                         s /*size_t*/);
  // End
}
