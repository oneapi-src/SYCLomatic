#include "curand_kernel.h"

__global__ void test() {
  double mean, stddev;
  // Start
  curandStatePhilox4_32_10_t *ps;
  curand_log_normal4_double(ps /*curandStatePhilox4_32_10_t **/,
                            mean /*double*/, stddev /*double*/);
  // End
}
