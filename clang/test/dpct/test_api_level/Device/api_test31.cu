// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test31_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test31_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test31_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test31_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test31_out


// CHECK: 18

// TEST_FEATURE: Device_pointer_attributes

#include<cuda.h>

int main() {
  void *h_A;
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, h_A);
  return 0;
}
