void test(CUdevice device, unsigned int flags) {
  // Start
  cuDevicePrimaryCtxSetFlags(device /*CUdevice*/, flags /*unsigned int*/);
  // End
}
