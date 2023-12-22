// Option: --use-dpcpp-extensions=peer_access
void test(int i, unsigned int u) {
  // Start
  cudaDeviceEnablePeerAccess(i /*int*/, u /*unsigned int*/);
  // End
}
