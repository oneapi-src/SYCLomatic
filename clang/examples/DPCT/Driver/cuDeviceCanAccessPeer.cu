void test(int *pi, CUdevice d1, CUdevice d2) {
  // Start
  cuDeviceCanAccessPeer(pi /*int **/, d1 /*CUdevice*/, d2 /*CUdevice*/);
  // End
}
