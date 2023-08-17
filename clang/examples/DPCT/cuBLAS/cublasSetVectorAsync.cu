#include "cublas_v2.h"

void test(int n, int elementsize, const void *from, int incx, void *to,
          int incy, cudaStream_t stream) {
  // Start
  cublasSetVectorAsync(n /*int*/, elementsize /*int*/, from /*const void **/,
                       incx /*int*/, to /*void **/, incy /*int*/,
                       stream /*cudaStream_t*/);
  // End
}
