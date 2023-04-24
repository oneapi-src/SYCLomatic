// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/Memory/api_test43_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test43_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test43_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test43_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test43_out

// CHECK: 19

// TEST_FEATURE: Memory_async_dpct_free

#include "cublas_v2.h"
#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t cusolverH;
  float B_f, C_f = 0;
  int devInfo;
  cusolverDnSpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, 0, 0, &C_f, 4, &B_f, 4, &devInfo);
  return 0;
}
