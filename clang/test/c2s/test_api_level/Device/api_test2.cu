// RUN: c2s --format-range=none  --use-custom-helper=api -out-root %T/Device/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test2_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test2_out

// CHECK: 13
// TEST_FEATURE: Device_cpu_device

int main() {
  int *devPtr;
  size_t count;
  cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
  return 0;
}
