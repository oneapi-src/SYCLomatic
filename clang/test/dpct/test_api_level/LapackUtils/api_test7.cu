// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test7_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test7_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test7_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test7_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test7_out

// CHECK: 33
// TEST_FEATURE: LapackUtils_getrs

#include "cusolverDn.h"

int main() {
  float* a_s;
  int64_t* ipiv_s;
  float* b_s;
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  int *info;

  cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_32F, a_s, 2, ipiv_s, CUDA_R_32F, b_s, 2, info);
  return 0;
}
