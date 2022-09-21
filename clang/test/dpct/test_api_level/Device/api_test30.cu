// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test30_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test30_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test30_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test30_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test30_out

// CHECK: 3
// TEST_FEATURE: Device_typedef_event_ptr
// TEST_FEATURE: Device_destroy_event

int main() {
  cudaEvent_t start;
  cudaEventCreate(&start);
  cudaEventDestroy(start);
  return 0;
}
