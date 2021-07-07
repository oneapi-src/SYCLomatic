// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test8_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test8_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test8_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test8_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test8_out

// CHECK: 9
// TEST_FEATURE: Memory_get_buffer_T

#include "cublas_v2.h"
#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t* cusolverH = NULL;
  float A_f = 0;
  float workspace_f = 0;
  int Lwork = 0;
  int devInfo = 0;
  cusolverDnSpotrf(*cusolverH, CUBLAS_FILL_MODE_LOWER, 10, &A_f, 10, &workspace_f, Lwork, &devInfo);
  return 0;
}
