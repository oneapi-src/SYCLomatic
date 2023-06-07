// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test15_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test15_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test15_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test15_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test15_out

// CHECK: 14
// TEST_FEATURE: Device_dev_mgr_device_count

#include "cuda.h"

int main() {
  int result1;
  cuDeviceGetCount(&result1);
  return 0;
}
