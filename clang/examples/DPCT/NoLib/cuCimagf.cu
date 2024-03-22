#include <cuComplex.h>

__global__ void test(cuFloatComplex c) {
  // Start
  cuCimagf(c /*cuFloatComplex*/);
  // End
}
