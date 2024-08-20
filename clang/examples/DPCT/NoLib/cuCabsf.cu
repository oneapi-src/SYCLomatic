#include <cuComplex.h>

__global__ void test(cuFloatComplex c) {
  // Start
  cuCabsf(c /*cuFloatComplex*/);
  // End
}
