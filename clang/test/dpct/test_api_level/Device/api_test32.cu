// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test32_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test32_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test32_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test32_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test32_out

#include <nccl.h>

// CHECK: 29
// TEST_FEATURE: CclUtils_typedef_ccl_comm_ptr
// TEST_FEATURE: Device_get_device_id
// TEST_FEATURE: CclUtils_ccl_init_helper
// TEST_FEATURE: CclUtils_communicator_wrapper_get_device
int main() {
  int  device;
  ncclComm_t comm;
  ncclCommCuDevice(comm, &device);
}