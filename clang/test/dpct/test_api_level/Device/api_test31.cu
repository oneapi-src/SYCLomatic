// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test29_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test29_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test29_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test29_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test29_out

// CHECK: 8
// TEST_FEATURE: Device_device_ext_get_work_item_max_dim_x_size
// TEST_FEATURE: Device_device_ext_get_work_item_max_dim_y_size
// TEST_FEATURE: Device_device_ext_get_work_item_max_dim_z_size

int main() {
  int result1;
  CUdevice device;
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device);
  return 0;
}
