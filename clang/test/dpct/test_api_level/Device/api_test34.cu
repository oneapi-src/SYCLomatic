// RUN: dpct --format-range=none --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test34_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test34_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test34_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test34_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test34_out

// CHECK: 19
// TEST_FEATURE: Device_has_capability_or_fail
#include "cuda_fp16.h"

__device__ void device_fp() {
  double a = 0;
  half b = 0;
}

__global__ void test_fp() { device_fp(); }

int main() {
  test_fp<<<1, 1>>>();
  return 0;
}
