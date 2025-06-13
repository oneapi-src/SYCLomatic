#include "curand_kernel.h"

__global__ void test() {
  curandStatePhilox4_32_10_t *ps;
  double mean, stddev;
  // Start
  curand_log_normal4_double(ps /*curandStatePhilox4_32_10_t **/,
                            mean /*double*/, stddev /*double*/);
  // End
}
