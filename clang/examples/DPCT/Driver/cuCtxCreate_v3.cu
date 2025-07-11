void test(CUcontext *ctx, CUexecAffinityParam *params_array, int num,
          unsigned int flags, CUdevice device) {
  // Start
  cuCtxCreate_v3(ctx /*CUcontext **/, params_array /*CUexecAffinityParam **/, num /*int*/, flags /*unsigned int*/, device /*CUdevice*/);
  // End
}