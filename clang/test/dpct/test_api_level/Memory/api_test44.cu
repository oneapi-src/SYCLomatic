// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test44_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test44_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test44_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test44_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test44_out

// CHECK: 17

// TEST_FEATURE: Memory_pointer_attributes

#include<cuda.h>

int main() {
  void *h_A;
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, h_A);
  return 0;
}
