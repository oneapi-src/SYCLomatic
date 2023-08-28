#include "curand.h"

void test(cudaStream_t s) {
  // Start
  curandGenerator_t g;
  curandSetStream(g /*curandGenerator_t*/, s /*cudaStream_t*/);
  // End
}
