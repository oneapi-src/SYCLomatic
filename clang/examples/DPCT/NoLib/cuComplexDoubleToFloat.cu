#include <cuComplex.h>

__global__ void test(cuDoubleComplex c) {
  // Start
  cuComplexDoubleToFloat(c /*cuDoubleComplex*/);
  // End
}
