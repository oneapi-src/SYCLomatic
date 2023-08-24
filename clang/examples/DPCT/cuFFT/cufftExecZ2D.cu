#include "cufft.h"

void test(cufftDoubleComplex *in, cufftDoubleReal *out) {
  // Start
  cufftHandle plan;
  cufftCreate(&plan /*cufftHandle **/);
  cufftExecZ2D(plan /*cufftHandle*/, in /*cufftDoubleComplex **/,
               out /*cufftDoubleReal **/);
  // End
}
