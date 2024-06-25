#include <cuComplex.h>

__global__ void test(cuDoubleComplex c1, cuDoubleComplex c2,
                     cuDoubleComplex c3) {
  // Start
  cuCfma(c1 /*cuDoubleComplex*/, c2 /*cuDoubleComplex*/,
         c3 /*cuDoubleComplex*/);
  // End
}
