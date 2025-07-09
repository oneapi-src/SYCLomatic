void test(CUcontext *ctx, CUexecAffinityParam *params_array, int num,
          unsigned int flags, CUdevice device) {
  // Start
  cuCtxCreate_v3(ctx, params_array, num, flags, device);
  // End
}