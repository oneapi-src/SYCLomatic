#include <cuComplex.h>

__global__ void test(double d1, double d2) {
  // Start
  make_cuDoubleComplex(d1 /*double*/, d2 /*double*/);
  // End
}
