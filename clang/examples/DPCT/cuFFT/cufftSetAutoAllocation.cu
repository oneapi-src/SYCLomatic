#include "cufft.h"

void test(cufftHandle plan, int autoallocate) {
  // Start
  cufftSetAutoAllocation(plan /*cufftHandle*/, autoallocate /*int*/);
  // End
}
