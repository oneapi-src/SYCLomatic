// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test19_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test19_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test19_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test19_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test19_out

// CHECK: 32
// TEST_FEATURE: LapackUtils_syhegvx_scratchpad_size

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigType_t itype;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int n;
  const float *A;
  int lda;
  const float *B;
  int ldb;
  float vl;
  float vu;
  int il;
  int iu;
  int *h_meig;
  const float *W;
  int *lwork;

  cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B,
                               ldb, vl, vu, il, iu, h_meig, W, lwork);
  return 0;
}
