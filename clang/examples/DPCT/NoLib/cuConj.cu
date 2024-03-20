#include <cuComplex.h>

__global__ void test(cuDoubleComplex c) {
  // Start
  cuConj(c /*cuDoubleComplex*/);
  // End
}
