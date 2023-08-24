#include "cufft.h"

void test(cufftComplex *in, cufftReal *out) {
  // Start
  cufftHandle plan;
  cufftCreate(&plan /*cufftHandle **/);
  cufftExecC2R(plan /*cufftHandle*/, in /*cufftComplex **/,
               out /*cufftReal **/);
  // End
}
