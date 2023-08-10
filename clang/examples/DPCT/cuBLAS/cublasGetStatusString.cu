#include "cublas_v2.h"

void test(const char *res, cublasStatus_t status) {
  // Start
  res /*const char **/ = cublasGetStatusString(status /*cublasStatus_t*/);
  // End
}
