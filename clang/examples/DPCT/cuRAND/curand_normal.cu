#include "curand_kernel.h"

__global__ void test() {
  // Start
  curandStateScrambledSobol64_t *ps1;
  curandStateSobol64_t *ps2;
  curandStateScrambledSobol32_t *ps3;
  curandStateSobol32_t *ps4;
  curandStateMtgp32_t *ps5;
  curandStateMRG32k3a_t *ps6;
  curandStatePhilox4_32_10_t *ps7;
  curandStateXORWOW_t *ps8;
  curand_normal(ps1 /*curandStateScrambledSobol64_t **/);
  curand_normal(ps2 /*curandStateSobol64_t **/);
  curand_normal(ps3 /*curandStateScrambledSobol32_t **/);
  curand_normal(ps4 /*curandStateSobol32_t **/);
  curand_normal(ps5 /*curandStateMtgp32_t **/);
  curand_normal(ps6 /*curandStateMRG32k3a_t **/);
  curand_normal(ps7 /*curandStatePhilox4_32_10_t **/);
  curand_normal(ps8 /*curandStateXORWOW_t **/);
  // End
}
