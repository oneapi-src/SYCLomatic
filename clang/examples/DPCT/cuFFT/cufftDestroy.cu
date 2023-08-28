#include "cufft.h"

void test(cufftHandle plan) {
  // Start
  cufftDestroy(plan /*cufftHandle*/);
  // End
}
