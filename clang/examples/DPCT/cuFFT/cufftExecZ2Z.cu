#include "cufft.h"

void test(cufftDoubleComplex *in, cufftDoubleComplex *out, int dir) {
  // Start
  cufftHandle plan;
  cufftCreate(&plan /*cufftHandle **/);
  cufftExecZ2Z(plan /*cufftHandle*/, in /*cufftDoubleComplex **/,
               out /*cufftDoubleComplex **/, dir /*int*/);
  // End
}
