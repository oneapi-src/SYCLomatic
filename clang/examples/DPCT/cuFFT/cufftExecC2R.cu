#include "cufft.h"

void test(cufftHandle plan, cufftComplex *in, cufftReal *out) {
  // Start
  cufftExecC2R(plan /*cufftHandle*/, in /*cufftComplex **/,
               out /*cufftReal **/);
  // End
}
