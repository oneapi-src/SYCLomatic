#include "cublas.h"

void foo_h() {
  int *a;
  cudaDeviceGetPCIBusId(nullptr, 0, 0);
  cudaMalloc(&a, sizeof(int) * 4);
}