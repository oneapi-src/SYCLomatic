#include "cufft.h"

void test(cufftHandle plan, int nx, int ny, cufftType type, size_t *worksize) {
  // Start
  cufftEstimate2d(nx /*int*/, ny /*int*/, type /*cufftType*/,
                  worksize /*size_t **/);
  // End
}
