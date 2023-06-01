// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test15_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test15_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test15_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test15_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test15_out

// CHECK: 34
// TEST_FEATURE: LapackUtils_syheevx_scratchpad_size_T

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int n;
  const float *A;
  int lda;
  float vl;
  float vu;
  int il;
  int iu;
  int *h_meig;
  const float *W;
  int *lwork;

  cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il,
                               iu, h_meig, W, lwork);
  return 0;
}
