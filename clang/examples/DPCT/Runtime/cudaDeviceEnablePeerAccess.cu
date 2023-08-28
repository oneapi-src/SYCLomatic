// Migration desc: The API is Removed.
void test(int i, unsigned int u) {
  // Start
  cudaDeviceEnablePeerAccess(i /*int*/, u /*unsigned int*/);
  // End
}
