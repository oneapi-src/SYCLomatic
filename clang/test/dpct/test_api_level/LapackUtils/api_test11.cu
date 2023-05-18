// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test11_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test11_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test11_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test11_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test11_out

// CHECK: 33
// TEST_FEATURE: LapackUtils_gesvd

#include "cusolverDn.h"

int main() {
  float* a_s;
  float* s_s;
  float* u_s;
  float* vt_s;
  cusolverDnHandle_t handle;
  size_t device_ws_size_s;
  size_t host_ws_size_s;
  cusolverDnParams_t params;
  void* device_ws_s;
  void* host_ws_s;
  int *info;

  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, s_s, CUDA_R_32F, u_s, 2, CUDA_R_32F, vt_s, 2, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  return 0;
}
