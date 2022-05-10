// RUN: echo "empty command"

#include "header.h"

int main () {
  cublasStatus_t s;
  cublasHandle_t handle;
  int N = 275;
  double *x1;
  int *result;

  cublasIdamax(handle, N, x1, N, result);
}
