#include "cufft.h"

void test(cufftHandle *plan, int nx, int ny, int nz, cufftType type) {
  // Start
  cufftPlan3d(plan /*cufftHandle **/, nx /*int*/, ny /*int*/, nz /*int*/,
              type /*cufftType*/);
  // End
}
