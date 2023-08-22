#include "cufft.h"

void test(cufftHandle *plan, int nx, int ny, cufftType type) {
  // Start
  cufftPlan2d(plan /*cufftHandle **/, nx /*int*/, ny /*int*/,
              type /*cufftType*/);
  // End
}
