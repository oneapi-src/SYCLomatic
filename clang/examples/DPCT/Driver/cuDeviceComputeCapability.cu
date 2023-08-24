void test(int *pi1, int *pi2, CUdevice d) {
  // Start
  cuDeviceComputeCapability(pi1 /*int **/, pi2 /*int **/, d /*CUdevice*/);
  // End
}
