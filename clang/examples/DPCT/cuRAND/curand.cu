#include "curand_kernel.h"

__global__ void test() {
  // Start
  curandStateMtgp32_t *ps1;
  curandStateScrambledSobol64_t *ps2;
  curandStateSobol64_t *ps3;
  curandStateScrambledSobol32_t *ps4;
  curandStateSobol32_t *ps5;
  curandStateMRG32k3a_t *ps6;
  curandStatePhilox4_32_10_t *ps7;
  curandStateXORWOW_t *ps8;
  curand(ps1 /*curandStateMtgp32_t **/);
  curand(ps2 /*curandStateScrambledSobol64_t **/);
  curand(ps3 /*curandStateSobol64_t **/);
  curand(ps4 /*curandStateScrambledSobol32_t **/);
  curand(ps5 /*curandStateSobol32_t **/);
  curand(ps6 /*curandStateMRG32k3a_t **/);
  curand(ps7 /*curandStatePhilox4_32_10_t **/);
  curand(ps8 /*curandStateXORWOW_t **/);
  // End
}
