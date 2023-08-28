#include "curand_kernel.h"

__global__ void test(float f1, float f2) {
  // Start
  curandStateMRG32k3a_t *ps1;
  curandStatePhilox4_32_10_t *ps2;
  curandStateXORWOW_t *ps3;
  curand_log_normal(ps1 /*curandStateMRG32k3a_t **/, f1 /*float*/,
                    f2 /*float*/);
  curand_log_normal(ps2 /*curandStatePhilox4_32_10_t **/, f1 /*float*/,
                    f2 /*float*/);
  curand_log_normal(ps3 /*curandStateXORWOW_t **/, f1 /*float*/, f2 /*float*/);
  // End
}
