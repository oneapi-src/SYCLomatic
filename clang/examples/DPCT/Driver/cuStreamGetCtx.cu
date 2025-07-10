#include <cuda.h>
void test(CUstream stream, CUcontext& context) {
  // Start
  cuStreamGetCtx(stream/*CUstream*/, &context/*CUcontext **/);
  // End
}