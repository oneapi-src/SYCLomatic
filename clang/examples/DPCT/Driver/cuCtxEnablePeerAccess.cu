void test(CUcontext c, unsigned u) {
  // Start
  cuCtxEnablePeerAccess(c /*CUcontext*/, u /*unsigned*/);
  // End
}
