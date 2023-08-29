#include "curand_kernel.h"

__global__ void test() {
  // Start
  curandStatePhilox4_32_10_t *ps;
  curand_uniform4(ps /*curandStatePhilox4_32_10_t **/);
  // End
}
