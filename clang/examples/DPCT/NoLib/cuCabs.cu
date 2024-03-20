#include <cuComplex.h>

__global__ void test(cuDoubleComplex c) {
  // Start
  cuCabs(c /*cuDoubleComplex*/);
  // End
}
