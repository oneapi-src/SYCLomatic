// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasMemory/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasMemory/api_test3_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasMemory/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasMemory/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasMemory/api_test3_out

// CHECK: 15
// TEST_FEATURE: DplExtrasMemory_free_device

#include <thrust/device_ptr.h>
#include <thrust/device_free.h>
int main() {
  double *a;
  thrust::device_ptr<double> d_ptr(a);
  thrust::device_free(d_ptr);
  return 0;
}
