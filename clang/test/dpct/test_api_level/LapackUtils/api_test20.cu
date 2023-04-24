// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test20_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test20_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test20_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test20_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test20_out

// CHECK: 38
// TEST_FEATURE: LapackUtils_syhegvx

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigType_t itype;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int n;
  float *A;
  int lda;
  float *B;
  int ldb;
  float vl;
  float vu;
  int il;
  int iu;
  int *h_meig;
  float *W;
  float *work;
  int lwork;
  int *devInfo;

  cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                    il, iu, h_meig, W, work, lwork, devInfo);
  return 0;
}
