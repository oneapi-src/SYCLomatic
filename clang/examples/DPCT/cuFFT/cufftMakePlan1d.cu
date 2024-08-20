#include "cufft.h"

void test(cufftHandle plan, int nx, cufftType type, int num_of_trans,
          size_t *worksize) {
  // Start
  cufftMakePlan1d(plan /*cufftHandle*/, nx /*int*/, type /*cufftType*/,
                  num_of_trans /*int*/, worksize /*size_t **/);
  // End
}
