#include "curand_kernel.h"

__global__ void test() {
  curandStatePhilox4_32_10_t *ps;
  // Start
  curand_uniform4_double(ps /*curandStatePhilox4_32_10_t **/);
  // End
}
