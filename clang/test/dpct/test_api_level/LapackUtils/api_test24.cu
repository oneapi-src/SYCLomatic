// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test24_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test24_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test24_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test24_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test24_out

// CHECK: 31
// TEST_FEATURE: LapackUtils_syheev_scratchpad_size

#include "cusolverDn.h"

int main() {
  float *a_s;
  float *w_s;
  cusolverDnHandle_t handle;
  syevjInfo_t params;
  int lwork_s;
  cusolverDnSsyevj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, w_s, &lwork_s, params);
  return 0;
}
