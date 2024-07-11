#include "cufft.h"

void test(cufftHandle plan, void *workspace) {
  // Start
  cufftSetWorkArea(plan /*cufftHandle*/, workspace /*void **/);
  // End
}
