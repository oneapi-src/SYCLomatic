#include "cufft.h"

void test(cufftReal *in, cufftComplex *out) {
  // Start
  cufftHandle plan;
  cufftCreate(&plan /*cufftHandle **/);
  cufftExecR2C(plan /*cufftHandle*/, in /*cufftReal **/,
               out /*cufftComplex **/);
  // End
}
