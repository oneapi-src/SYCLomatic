// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test26_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test26_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test26_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test26_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test26_out

// CHECK: 18

// TEST_FEATURE: Util_pointer_attributes

#include<cuda.h>

int main() {
  void *h_A;
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, h_A);
  return 0;
}
