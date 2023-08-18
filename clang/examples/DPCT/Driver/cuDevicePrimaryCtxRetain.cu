void test(CUcontext *pc, CUdevice d) {
  // Start
  cuDevicePrimaryCtxRetain(pc /*CUcontext **/, d /*CUdevice*/);
  // End
}
