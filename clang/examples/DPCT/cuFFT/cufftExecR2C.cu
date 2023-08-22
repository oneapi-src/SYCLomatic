#include "cufft.h"

void test(cufftHandle plan, cufftReal *in, cufftComplex *out) {
  // Start
  cufftExecR2C(plan /*cufftHandle*/, in /*cufftReal **/,
               out /*cufftComplex **/);
  // End
}
