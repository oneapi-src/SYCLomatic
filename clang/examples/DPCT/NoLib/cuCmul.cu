#include <cuComplex.h>

__global__ void test(cuDoubleComplex c1, cuDoubleComplex c2) {
  // Start
  cuCmul(c1 /*cuDoubleComplex*/, c2 /*cuDoubleComplex*/);
  // End
}
