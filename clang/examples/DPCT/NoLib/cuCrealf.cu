#include <cuComplex.h>

__global__ void test(cuFloatComplex c) {
  // Start
  cuCrealf(c /*cuFloatComplex*/);
  // End
}
