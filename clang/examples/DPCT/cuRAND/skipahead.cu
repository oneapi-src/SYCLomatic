#include "curand_kernel.h"

__global__ void test(unsigned long long ull, unsigned int u) {
  // Start
  curandStateMRG32k3a_t *ps1;
  curandStatePhilox4_32_10_t *ps2;
  curandStateXORWOW_t *ps3;
  skipahead(ull, ps1 /*curandStateMRG32k3a_t **/);
  skipahead(ull, ps2 /*curandStatePhilox4_32_10_t **/);
  skipahead(ull, ps3 /*curandStateXORWOW_t **/);
  skipahead(u, ps1 /*curandStateMRG32k3a_t **/);
  skipahead(u, ps2 /*curandStatePhilox4_32_10_t **/);
  skipahead(u, ps3 /*curandStateXORWOW_t **/);
  // End
}
