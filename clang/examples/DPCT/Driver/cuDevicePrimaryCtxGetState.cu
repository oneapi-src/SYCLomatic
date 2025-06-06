void test(CUdevice device, unsigned int flags, int active) {
  // Start
  cuDevicePrimaryCtxGetState(device /*CUdevice*/, &flags /*unsigned int **/,
                             &active /*int **/);
  // End
}
