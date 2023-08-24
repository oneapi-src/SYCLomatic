#include "cufft.h"

void test(int nx, int ny, int nz, cufftType type) {
  // Start
  cufftHandle plan;
  cufftPlan3d(&plan /*cufftHandle **/, nx /*int*/, ny /*int*/, nz /*int*/,
              type /*cufftType*/);
  // End
}
