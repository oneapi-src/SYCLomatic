#include "curand_kernel.h"

__global__ void test(unsigned long long ull1, unsigned long long ull2,
                     unsigned long long ull3, curandStateMRG32k3a_t *ps1,
                     curandStatePhilox4_32_10_t *ps2,
                     curandStateXORWOW_t *ps3) {
  // Start
  curand_init(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/,
              ull3 /*unsigned long long*/, ps1 /*curandStateMRG32k3a_t **/);
  curand_init(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/,
              ull3 /*unsigned long long*/,
              ps2 /*curandStatePhilox4_32_10_t **/);
  curand_init(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/,
              ull3 /*unsigned long long*/, ps3 /*curandStateXORWOW_t **/);
  // End
}
