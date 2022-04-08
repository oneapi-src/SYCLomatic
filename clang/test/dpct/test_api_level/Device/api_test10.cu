// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test10_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test10_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test10_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test10_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test10_out

// CHECK: 16
// TEST_FEATURE: Device_device_ext_set_saved_queue

#include "cublas.h"

int main() {
  cudaStream_t stream;
  cublasSetKernelStream(stream);
  return 0;
}
