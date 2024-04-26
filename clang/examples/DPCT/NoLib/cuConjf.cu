#include <cuComplex.h>

__global__ void test(cuFloatComplex c) {
  // Start
  cuConjf(c /*cuFloatComplex*/);
  // End
}
