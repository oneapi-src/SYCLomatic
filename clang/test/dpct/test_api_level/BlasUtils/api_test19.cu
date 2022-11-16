// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test19_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test19_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test19_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test19_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test19_out

// CHECK: 24

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_rot

int main() {
  cublasHandle_t handle;
  void *x;
  void *y;
  void *sin;
  void *cos;

  cublasRotEx(handle, 4, x, CUDA_R_32F, 1,  y, CUDA_R_32F, 1,  cos, sin, CUDA_R_32F, CUDA_R_32F);
  return 0;
}
