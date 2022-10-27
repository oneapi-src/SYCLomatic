// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/DplExtrasMemory/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasMemory/api_test2_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasMemory/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasMemory/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasMemory/api_test2_out

// CHECK: 24
// TEST_FEATURE: DplExtrasMemory_device_ptr
// TEST_FEATURE: DplExtrasMemory_device_pointer_forward_decl

#include <thrust/device_ptr.h>

int main() {
  double *p;
  thrust::device_ptr<double> dp(p);
  return 0;
}
