#include "cufft.h"

void test(int nx, cufftType type, int num_of_trans) {
  // Start
  cufftHandle plan;
  cufftPlan1d(&plan /*cufftHandle **/, nx /*int*/, type /*cufftType*/,
              num_of_trans /*int*/);
  // End
}
