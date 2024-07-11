#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, double *x, int64_t incx, double *y,
          int64_t incy, const double *c, const double *s) {
  // Start
  cublasDrot_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*double **/,
                incx /*int64_t*/, y /*double **/, incy /*int64_t*/,
                c /*const double **/, s /*const double **/);
  // End
}
