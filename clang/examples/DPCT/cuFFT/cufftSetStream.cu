#include "cufft.h"

void test(cufftHandle plan, cudaStream_t s) {
  // Start
  cufftSetStream(plan /*cufftHandle*/, s /*cudaStream_t*/);
  // End
}
