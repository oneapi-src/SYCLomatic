#include <cuComplex.h>

__global__ void test(float f1, float f2) {
  // Start
  make_cuComplex(f1 /*float*/, f2 /*float*/);
  // End
}
