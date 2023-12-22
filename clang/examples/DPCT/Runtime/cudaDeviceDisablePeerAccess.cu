// Option: --use-dpcpp-extensions=peer_access
void test(int i) {
  // Start
  cudaDeviceDisablePeerAccess(i /*int*/);
  // End
}
