// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/DplUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/DplUtils/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplUtils/api_test1_out

// CHECK: 2
// TEST_FEATURE: DplUtils_non_local_include_dependency

#include <thrust/device_ptr.h>

int main() {
  return 0;
}
