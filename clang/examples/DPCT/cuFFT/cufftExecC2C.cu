#include "cufft.h"

void test(cufftComplex *in, cufftComplex *out, int dir) {
  // Start
  cufftHandle plan;
  cufftCreate(&plan /*cufftHandle **/);
  cufftExecC2C(plan /*cufftHandle*/, in /*cufftComplex **/,
               out /*cufftComplex **/, dir /*int*/);
  // End
}
