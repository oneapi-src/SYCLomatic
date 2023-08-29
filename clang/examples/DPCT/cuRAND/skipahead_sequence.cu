#include "curand_kernel.h"

__global__ void test(unsigned long long ull) {
  // Start
  curandStateMRG32k3a_t *ps1;
  curandStatePhilox4_32_10_t *ps2;
  curandStateXORWOW_t *ps3;
  skipahead_sequence(ull, ps1 /*curandStateMRG32k3a_t **/);
  skipahead_sequence(ull, ps2 /*curandStatePhilox4_32_10_t **/);
  skipahead_sequence(ull, ps3 /*curandStateXORWOW_t **/);
  // End
}
