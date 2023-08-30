#include "curand_kernel.h"

__global__ void test(float f1, float f2) {
  // Start
  curandStatePhilox4_32_10_t *ps;
  curand_log_normal4(ps /*curandStatePhilox4_32_10_t **/, f1 /*float*/,
                     f2 /*float*/);
  // End
}
