void test(CUcontext *ctx, CUctxCreateParams *params_array, unsigned int flags,
          CUdevice device) {
  // Start
  cuCtxCreate_v4(ctx, params_array, flags, device);
  // End
}