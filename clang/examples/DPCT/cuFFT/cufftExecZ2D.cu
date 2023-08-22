#include "cufft.h"

void test(cufftHandle plan, cufftDoubleComplex *in, cufftDoubleReal *out) {
  // Start
  cufftExecZ2D(plan /*cufftHandle*/, in /*cufftDoubleComplex **/,
               out /*cufftDoubleReal **/);
  // End
}
