// RUN: dpct --use-custom-helper=file  --use-explicit-namespace=none  -out-root %T/out2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/out2/include/dpct/atomic.hpp --match-full-lines %S/atomic_ref.txt
// RUN: FileCheck --input-file %T/out2/include/dpct/blas_utils.hpp --match-full-lines %S/blas_utils_ref.txt

#include "cublas_v2.h"

__global__ void test(int *data) {
  int tid = threadIdx.x;
  atomicAdd(&data[0], tid);
}

int main() {
  cublasHandle_t handle;
  int n = 275;
  int lda = 275;

  float **Aarray_S = 0;
  int *PivotArray = 0;
  int *infoArray = 0;
  int batchSize = 10;

  cublasSgetrfBatched(handle, n, Aarray_S, lda, PivotArray, infoArray, batchSize);
  return 0;
}
