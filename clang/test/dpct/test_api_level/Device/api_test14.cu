// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test14_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test14_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test14_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test14_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test14_out

// CHECK: 15
// TEST_FEATURE: Device_dev_mgr_current_device_id

int main() {
  int dev_id;
  cudaGetDevice(&dev_id);
  return 0;
}
