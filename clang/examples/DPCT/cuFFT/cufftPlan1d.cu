#include "cufft.h"

void test(cufftHandle *plan, int nx, cufftType type, int num_of_trans) {
  // Start
  cufftPlan1d(plan /*cufftHandle **/, nx /*int*/, type /*cufftType*/,
              num_of_trans /*int*/);
  // End
}
