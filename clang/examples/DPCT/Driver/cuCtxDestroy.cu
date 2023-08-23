// Migration desc: The API is Removed.
void test(CUcontext c) {
  // Start
  cuCtxDestroy(c /*CUcontext*/);
  // End
}
