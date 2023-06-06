// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test23_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test23_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test23_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test23_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test23_out

// CHECK: 33
// TEST_FEATURE: LapackUtils_syheev

#include "cusolverDn.h"

int main() {
  float *a_s;
  float *w_s;
  cusolverDnHandle_t handle;
  syevjInfo_t params;
  int lwork_s;
  float *device_ws_s;
  int *info;
  cusolverDnSsyevj(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, w_s, device_ws_s, lwork_s, info, params);
  return 0;
}
