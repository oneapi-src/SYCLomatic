#include "cufft.h"

void test(cufftDoubleReal *in, cufftDoubleComplex *out) {
  // Start
  cufftHandle plan;
  cufftCreate(&plan /*cufftHandle **/);
  cufftExecD2Z(plan /*cufftHandle*/, in /*cufftDoubleReal **/,
               out /*cufftDoubleComplex **/);
  // End
}
