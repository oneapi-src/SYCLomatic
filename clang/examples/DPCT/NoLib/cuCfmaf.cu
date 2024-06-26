#include <cuComplex.h>

__global__ void test(cuFloatComplex c1, cuFloatComplex c2, cuFloatComplex c3) {
  // Start
  cuCfmaf(c1 /*cuFloatComplex*/, c2 /*cuFloatComplex*/, c3 /*cuFloatComplex*/);
  // End
}
