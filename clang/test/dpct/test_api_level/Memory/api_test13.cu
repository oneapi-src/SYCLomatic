// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test13_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test13_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test13_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test13_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test13_out

// CHECK: 12
// TEST_FEATURE: Memory_is_device_ptr

#include "cublas_v2.h"

int main() {
  cublasHandle_t handle;
  float *x_S;
  int *result;
  cublasIsamax(handle, 10, x_S, 1, result);
  return 0;
}
