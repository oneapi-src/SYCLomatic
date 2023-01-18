// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test5_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test5_out

// CHECK: 29
// TEST_FEATURE: LapackUtils_getrf_scratchpad_size

#include "cusolverDn.h"

int main() {
  float* a_s;
  int64_t* ipiv_s;
  cusolverDnHandle_t handle;
  size_t device_ws_size_s;
  size_t host_ws_size_s;
  cusolverDnParams_t params;

  cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  return 0;
}
