// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test27_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test27_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test27_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test27_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test27_out

// CHECK: 32
// TEST_FEATURE: LapackUtils_trtri

#include "cusolverDn.h"

int main() {
  float *a_s;
  cusolverDnHandle_t handle;
  size_t lwork_s;
  size_t lwork_host_s;
  void* device_ws_s;
  void* host_ws_s;
  int *info;
  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_32F, a_s, 2, device_ws_s, lwork_s, host_ws_s, lwork_host_s, info);
  return 0;
}
