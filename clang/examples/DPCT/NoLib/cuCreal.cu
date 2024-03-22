#include <cuComplex.h>

__global__ void test(cuDoubleComplex c) {
  // Start
  cuCreal(c /*cuDoubleComplex*/);
  // End
}
