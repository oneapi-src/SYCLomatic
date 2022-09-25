// RUN: dpct --format-range=none --no-dpcpp-extensions=device_info --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test13_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test13_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test13_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test13_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test13_out

// CHECK: 34
// TEST_FEATURE: Device_device_info_get_global_mem_size
// TEST_FEATURE: Device_get_current_device

int main() {
  size_t result1, result2;
  cuMemGetInfo(&result1, &result2);
  return 0;
}
