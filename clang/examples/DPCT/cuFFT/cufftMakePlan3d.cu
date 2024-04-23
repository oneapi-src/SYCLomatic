#include "cufft.h"

void test(cufftHandle plan, int nx, int ny, int nz, cufftType type,
          size_t *worksize) {
  // Start
  cufftMakePlan3d(plan /*cufftHandle*/, nx /*int*/, ny /*int*/, nz /*int*/,
                  type /*cufftType*/, worksize /*size_t **/);
  // End
}
