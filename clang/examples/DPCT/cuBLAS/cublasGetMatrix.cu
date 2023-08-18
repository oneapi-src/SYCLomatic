#include "cublas_v2.h"

void test(int rows, int cols, int elementsize, const void *a, int lda, void *b,
          int ldb) {
  // Start
  cublasGetMatrix(rows /*int*/, cols /*int*/, elementsize /*int*/,
                  a /*const void **/, lda /*int*/, b /*void **/, ldb /*int*/);
  // End
}
