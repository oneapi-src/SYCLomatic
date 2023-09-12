#include "curand_kernel.h"

__global__ void test(double d) {
  // Start
  curandStateMRG32k3a_t *ps1;
  curandStatePhilox4_32_10_t *ps2;
  curandStateXORWOW_t *ps3;
  curand_poisson(ps1 /*curandStateMRG32k3a_t **/, d /*double*/);
  curand_poisson(ps2 /*curandStatePhilox4_32_10_t **/, d /*double*/);
  curand_poisson(ps3 /*curandStateXORWOW_t **/, d /*double*/);
  // End
}
