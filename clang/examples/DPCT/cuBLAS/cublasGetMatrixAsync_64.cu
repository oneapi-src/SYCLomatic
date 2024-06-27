#include "cublas_v2.h"

void test(int64_t rows, int64_t cols, int64_t elementsize, const void *a,
          int64_t lda, void *b, int64_t ldb, cudaStream_t stream) {
  // Start
  cublasGetMatrixAsync_64(rows /*int64_t*/, cols /*int64_t*/,
                          elementsize /*int64_t*/, a /*const void **/,
                          lda /*int64_t*/, b /*void **/, ldb /*int64_t*/,
                          stream /*cudaStream_t*/);
  // End
}
