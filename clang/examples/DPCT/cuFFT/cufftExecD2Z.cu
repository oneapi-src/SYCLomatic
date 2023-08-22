#include "cufft.h"

void test(cufftHandle plan, cufftDoubleReal *in, cufftDoubleComplex *out) {
  // Start
  cufftExecD2Z(plan /*cufftHandle*/, in /*cufftDoubleReal **/,
               out /*cufftDoubleComplex **/);
  // End
}
