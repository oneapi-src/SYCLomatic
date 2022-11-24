// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test3_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test3_out

// CHECK: 22
// TEST_FEATURE: LapackUtils_potrf_batch

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  float ** a_s_ptrs;
  int *infoArray;
  cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, a_s_ptrs, 3, infoArray, 2);
  return 0;
}
