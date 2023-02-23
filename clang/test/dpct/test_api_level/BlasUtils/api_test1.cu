// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/BlasUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test1_out/count.txt --match-full-lines %s -check-prefix=FEATURE_NUMBER
// RUN: FileCheck --input-file %T/BlasUtils/api_test1_out/api_test1.dp.cpp --match-full-lines %s -check-prefix=CODE
// RUN: rm -rf %T/BlasUtils/api_test1_out

// FEATURE_NUMBER: 23

// CODE: // AAA
// CODE-NEXT:#include <sycl/sycl.hpp>
// CODE-NEXT:#include <dpct/dpct.hpp>
// CODE-NEXT:#include <oneapi/mkl.hpp>
// CODE-NEXT:// BBB

// AAA
#include "cublas_v2.h"
// BBB

// TEST_FEATURE: BlasUtils_geqrf_batch_wrapper
// TEST_FEATURE: BlasUtils_non_local_include_dependency

int main() {
  cublasHandle_t handle;
  int n = 275;
  int m = 275;
  int lda = 275;

  float **Aarray_S = 0;
  float **TauArray_S = 0;
  int *infoArray = 0;
  int batchSize = 10;

  cublasSgeqrfBatched(handle, m, n, Aarray_S, lda, TauArray_S, infoArray, batchSize);
  return 0;
}
