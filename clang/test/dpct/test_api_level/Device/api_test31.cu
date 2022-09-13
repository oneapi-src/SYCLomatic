// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test29_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test29_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test29_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test29_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test29_out

// CHECK: 1
#include "cuda_runtime.h"
// TEST_FEATURE: Device_device_info_get_max_stream_buffer_size
int main() {
  size_t result1;
  cuCtxGetLimit(&result1, CU_LIMIT_PRINTF_FIFO_SIZE);
  return 0;
}
