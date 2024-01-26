#include "cufft.h"
#include "cufftXt.h"

void test(cufftHandle plan, void *in, void *out, int dir) {
  // Start
  cufftXtExec(plan /*cufftHandle*/, in /*void **/, out /*void **/, dir /*int*/);
  // End
}
