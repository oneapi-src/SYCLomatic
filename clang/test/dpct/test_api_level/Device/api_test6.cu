// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test6_out


// CHECK: 39
// TEST_FEATURE: Device_device_ext_get_device_info_return_info
// TEST_FEATURE: Device_device_info_get_device_id
// TEST_FEATURE: Device_device_info_set_device_id
// TEST_FEATURE: Device_device_info_get_uuid
// TEST_FEATURE: Device_device_info_set_uuid
// TEST_FEATURE: Device_device_info_get_name
// TEST_FEATURE: Device_device_info_set_name

int main() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  deviceProp.pciDeviceID;
  deviceProp.name;
  return 0;
}
