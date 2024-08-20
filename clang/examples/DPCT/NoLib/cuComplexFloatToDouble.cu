#include <cuComplex.h>

__global__ void test(cuFloatComplex c) {
  // Start
  cuComplexFloatToDouble(c /*cuFloatComplex*/);
  // End
}
