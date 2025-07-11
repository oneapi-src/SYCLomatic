void test(CUcontext *ctx, CUctxCreateParams *params_array, unsigned int flags,
          CUdevice device) {
  // Start
  cuCtxCreate_v4(ctx /*CUcontext **/, params_array /*CUctxCreateParams **/, flags /*unsigned int*/, device /*CUdevice*/);
  // End
}