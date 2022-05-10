#include "cublas_v2.h"

int foo () {
  cublasStatus_t s;
  cublasHandle_t handle;
  int N = 275;
  float *x1;
  int *result;
  cublasIsamax(handle, N, x1, N, result);
}
