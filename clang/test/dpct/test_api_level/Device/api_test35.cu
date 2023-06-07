// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/Device/api_test35_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test35_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test35_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test35_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test35_out

#include "cuda.h"
// CHECK: 21
int main() {

  int dev = 0;
  cudaSetDevice(dev);
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);
  return 0;
}
