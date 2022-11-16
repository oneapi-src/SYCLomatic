// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test4_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test4_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test4_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test4_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test4_out

// CHECK: 40

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_getrf_batch_wrapper

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
