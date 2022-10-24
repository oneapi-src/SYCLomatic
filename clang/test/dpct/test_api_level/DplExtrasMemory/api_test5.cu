// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasMemory/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasMemory/api_test5_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasMemory/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasMemory/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasMemory/api_test5_out

// CHECK: 16
// TEST_FEATURE: DplExtrasMemory_malloc_device

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
int main() {
  thrust::device_malloc<double>(1);
  return 0;
}
