// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test5_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test5_out

// CHECK: 17
// TEST_FEATURE: Device_device_ext_destroy_queue

int main() {
  cudaStream_t s;
  cudaStreamDestroy(s);
  return 0;
}
