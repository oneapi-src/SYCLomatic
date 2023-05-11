// RUN: dpct --format-range=none --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test33_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test33_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test33_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test33_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test33_out

// CHECK: 15
// TEST_FEATURE: Device_device_ext_get_global_mem_size
// TEST_FEATURE: Device_device_ext_get_max_sub_group_size
// TEST_FEATURE: Device_device_ext_get_max_work_group_size
// TEST_FEATURE: Device_device_ext_get_mem_base_addr_align
// TEST_FEATURE: Device_device_ext_get_max_register_size_per_work_group
#include <cuda.h>
#include <cuda_runtime_api.h>

int main() {
  CUdevice device;
  int result0, result1, result2, result3, result4;
  cuDeviceGetAttribute(&result0, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
  cuDeviceGetAttribute(&result2, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
  cuDeviceGetAttribute(&result3, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device);
  cuDeviceGetAttribute(&result4, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device);
  return 0;
}
