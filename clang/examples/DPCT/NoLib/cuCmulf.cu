#include <cuComplex.h>

__global__ void test(cuFloatComplex c1, cuFloatComplex c2) {
  // Start
  cuCmulf(c1 /*cuFloatComplex*/, c2 /*cuFloatComplex*/);
  // End
}
