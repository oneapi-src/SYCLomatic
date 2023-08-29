#include "curand_kernel.h"

__global__ void test(double d1, double d2) {
  // Start
  curandStateMRG32k3a_t *ps1;
  curandStatePhilox4_32_10_t *ps2;
  curandStateXORWOW_t *ps3;
  curand_log_normal2_double(ps1 /*curandStateMRG32k3a_t **/, d1 /*double*/,
                            d2 /*double*/);
  curand_log_normal2_double(ps2 /*curandStatePhilox4_32_10_t **/, d1 /*double*/,
                            d2 /*double*/);
  curand_log_normal2_double(ps3 /*curandStateXORWOW_t **/, d1 /*double*/,
                            d2 /*double*/);
  // End
}
