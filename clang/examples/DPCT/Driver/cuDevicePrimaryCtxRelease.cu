// Migration desc: The API is Removed.
void test(CUdevice d) {
  // Start
  cuDevicePrimaryCtxRelease(d /*CUdevice*/);
  // End
}
