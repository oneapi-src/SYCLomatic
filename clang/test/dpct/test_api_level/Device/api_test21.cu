// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test21_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test21_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test21_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test21_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test21_out

// CHECK: 13
// TEST_FEATURE: Device_device_ext_is_native_atomic_supported

int main() {
  int res;
  cudaDeviceGetAttribute(&res, cudaDevAttrHostNativeAtomicSupported, 0);
  return 0;
}
