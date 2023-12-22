// Option: --use-dpcpp-extensions=peer_access
void test(int *pi, int i1, int i2) {
  // Start
  cudaDeviceCanAccessPeer(pi /*int **/, i1 /*int*/, i2 /*int*/);
  // End
}
