#include "curand_kernel.h"

__global__ void test(double d) {
  // Start
  curandStateScrambledSobol64_t *ps1;
  curandStateSobol64_t *ps2;
  curandStateScrambledSobol32_t *ps3;
  curandStateSobol32_t *ps4;
  curandStateMtgp32_t *ps5;
  curandStateMRG32k3a_t *ps6;
  curandStatePhilox4_32_10_t *ps7;
  curandStateXORWOW_t *ps8;
  curand_poisson(ps1 /*curandStateScrambledSobol64_t **/, d /*double*/);
  curand_poisson(ps2 /*curandStateSobol64_t **/, d /*double*/);
  curand_poisson(ps3 /*curandStateScrambledSobol32_t **/, d /*double*/);
  curand_poisson(ps4 /*curandStateSobol32_t **/, d /*double*/);
  curand_poisson(ps5 /*curandStateMtgp32_t **/, d /*double*/);
  curand_poisson(ps6 /*curandStateMRG32k3a_t **/, d /*double*/);
  curand_poisson(ps7 /*curandStatePhilox4_32_10_t **/, d /*double*/);
  curand_poisson(ps8 /*curandStateXORWOW_t **/, d /*double*/);
  // End
}
