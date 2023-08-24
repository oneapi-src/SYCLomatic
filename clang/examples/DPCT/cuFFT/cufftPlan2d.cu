#include "cufft.h"

void test(int nx, int ny, cufftType type) {
  // Start
  cufftHandle plan;
  cufftPlan2d(&plan /*cufftHandle **/, nx /*int*/, ny /*int*/,
              type /*cufftType*/);
  // End
}
