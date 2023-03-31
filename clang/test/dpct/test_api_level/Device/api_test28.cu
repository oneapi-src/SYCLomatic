// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test28_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test28_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test28_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test28_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test28_out

// CHECK: 36
// TEST_FEATURE: Device_device_info_get_global_mem_size
// TEST_FEATURE: Device_get_device
// TEST_FEATURE: Device_device_ext_get_device_info_return_info

#include "cuda.h"

int main() {
  size_t size;
  CUdevice device;
  cuDeviceTotalMem(&size, device);
  return 0;
}
