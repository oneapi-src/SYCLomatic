// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test16_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test16_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test16_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test16_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test16_out

// CHECK: 37
// TEST_FEATURE: LapackUtils_syheevx_T

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int n;
  float *A;
  int lda;
  float vl;
  float vu;
  int il;
  int iu;
  int *h_meig;
  float *W;
  float *work;
  int lwork;
  int *devInfo;

  cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                    h_meig, W, work, lwork, devInfo);
  return 0;
}
