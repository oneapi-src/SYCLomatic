// Migration desc: The API is Removed because this it is redundant in SYCL.
void test(CUdevice d) {
  // Start
  cuDevicePrimaryCtxRelease(d /*CUdevice*/);
  // End
}
