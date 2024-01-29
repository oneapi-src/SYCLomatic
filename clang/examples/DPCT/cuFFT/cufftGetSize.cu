#include "cufft.h"

void test(cufftHandle plan, size_t *worksize) {
  // Start
  cufftGetSize(plan /*cufftHandle*/, worksize /*size_t **/);
  // End
}
