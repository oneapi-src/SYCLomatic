void test(CUdeviceptr *pd, void *pv, unsigned int u) {
  // Start
  cuMemHostGetDevicePointer(pd /*CUdeviceptr **/, pv /*void **/,
                            u /*unsigned int*/);
  // End
}
