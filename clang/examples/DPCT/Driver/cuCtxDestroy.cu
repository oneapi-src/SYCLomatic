// Migration desc: The API is Removed because this it is redundant in SYCL.
void test(CUcontext c) {
  // Start
  cuCtxDestroy(c /*CUcontext*/);
  // End
}
