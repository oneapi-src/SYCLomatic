// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test1_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test1_out

// CHECK: 15
// TEST_FEATURE: Device_dev_mgr_get_device
// TEST_FEATURE: Device_device_ext_get_major_version
// TEST_FEATURE: Device_device_ext_get_minor_version

int main() {
  int result1, result2;
  CUdevice device;
  cuDeviceComputeCapability(&result1, &result2, device);
  return 0;
}
