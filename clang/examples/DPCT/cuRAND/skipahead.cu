#include "curand_kernel.h"

__global__ void test(unsigned long long ull, unsigned int u) {
  // Start
  curandStateMRG32k3a_t *ps1;
  curandStatePhilox4_32_10_t *ps2;
  curandStateXORWOW_t *ps3;
  /* 1 */ skipahead(ull, ps1 /*curandStateMRG32k3a_t **/);
  /* 2 */ skipahead(ull, ps2 /*curandStatePhilox4_32_10_t **/);
  /* 3 */ skipahead(ull, ps3 /*curandStateXORWOW_t **/);
  /* 4 */ skipahead(u, ps1 /*curandStateMRG32k3a_t **/);
  /* 5 */ skipahead(u, ps2 /*curandStatePhilox4_32_10_t **/);
  /* 6 */ skipahead(u, ps3 /*curandStateXORWOW_t **/);
  // End
}
