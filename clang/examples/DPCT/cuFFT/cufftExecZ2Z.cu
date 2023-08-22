#include "cufft.h"

void test(cufftHandle plan, cufftDoubleComplex *in, cufftDoubleComplex *out,
          int dir) {
  // Start
  cufftExecZ2Z(plan /*cufftHandle*/, in /*cufftDoubleComplex **/,
               out /*cufftDoubleComplex **/, dir /*int*/);
  // End
}
