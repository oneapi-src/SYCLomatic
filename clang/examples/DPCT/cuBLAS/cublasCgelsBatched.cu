#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
          int nrhs, cuComplex *const a_array[], int lda,
          cuComplex *const c_array[], int ldc, int *info, int *dev_info_array,
          int batch_size) {
  // Start
  cublasCgelsBatched(
      handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
      n /*int*/, nrhs /*int*/, a_array /*cuComplex *const []*/, lda /*int*/,
      c_array /*cuComplex *const []*/, ldc /*int*/, info /*int **/,
      dev_info_array /*int **/, batch_size /*int*/);
  // End
}
