// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test27_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test27_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test27_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test27_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test27_out

// CHECK: 18
// TEST_FEATURE: Device_select_device
// TEST_FEATURE: Device_get_current_device_id
int main() {
    CUdevice device;
    CUcontext ctx;
    cuCtxCreate(&ctx, CU_CTX_LMEM_RESIZE_TO_MAX, device);
    cuCtxGetDevice(&device);
    return 0;
}
