#include "cufft.h"

void test(cufftHandle plan, cufftComplex *in, cufftComplex *out, int dir) {
  // Start
  cufftExecC2C(plan /*cufftHandle*/, in /*cufftComplex **/,
               out /*cufftComplex **/, dir /*int*/);
  // End
}
