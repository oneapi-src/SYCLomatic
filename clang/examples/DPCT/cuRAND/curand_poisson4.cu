#include "curand_kernel.h"

__global__ void test(double d) {
  // Start
  curandStatePhilox4_32_10_t *ps;
  curand_poisson4(ps /*curandStatePhilox4_32_10_t **/, d /*double*/);
  // End
}
