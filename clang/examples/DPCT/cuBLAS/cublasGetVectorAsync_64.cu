#include "cublas_v2.h"

void test(int64_t n, int64_t elementsize, const void *from, int64_t incx,
          void *to, int64_t incy, cudaStream_t stream) {
  // Start
  cublasGetVectorAsync_64(n /*int64_t*/, elementsize /*int64_t*/,
                          from /*const void **/, incx /*int64_t*/,
                          to /*void **/, incy /*int64_t*/,
                          stream /*cudaStream_t*/);
  // End
}
