// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test22_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test22_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test22_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test22_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test22_out

// CHECK: 40
// TEST_FEATURE: Device_device_ext_get_max_compute_units

int main() {
  int res;
  cudaDeviceGetAttribute(&res, cudaDevAttrMultiProcessorCount, 0);
  return 0;
}
