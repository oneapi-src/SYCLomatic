// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Math/api_test9_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Math/api_test9_out/MainSourceFiles.yaml | wc -l > %T/Math/api_test9_out/count.txt
// RUN: FileCheck --input-file %T/Math/api_test9_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Math/api_test9_out

// CHECK: 9
// TEST_FEATURE: Math_compare
// TEST_FEATURE: Math_unordered_compare
// TEST_FEATURE: Math_compare_both
// TEST_FEATURE: Math_unordered_compare_both
// TEST_FEATURE: Math_compare2
// TEST_FEATURE: Math_unordered_compare2
// TEST_FEATURE: Math_isnan

#include "cuda_fp16.h"

__device__ void foo() {
  __half h, h_1;
  __half2 h2, h2_1;
  __hne(h, h_1);
  __hequ(h, h_1);
  __hbeq2(h2, h2_1);
  __hbequ2(h2, h2_1);
  __heq2(h2, h2_1);
  __hequ2(h2, h2_1);
  __hisnan2(h2);
}

int main() {
  return 0;
}
