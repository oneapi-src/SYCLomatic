#include "curand_kernel.h"

__global__ void test(unsigned long long ull) {
  // Start
  curandStateMRG32k3a_t *ps;
  skipahead_subsequence(ull, ps /*curandStateMRG32k3a_t **/);
  // End
}
