#include <cuComplex.h>

__global__ void test(cuDoubleComplex c) {
  // Start
  cuCimag(c /*cuDoubleComplex*/);
  // End
}
