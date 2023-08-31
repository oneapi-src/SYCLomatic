void test(CUdeviceptr *pDev, void *pHost, unsigned int u) {
  // Start
  cuMemHostGetDevicePointer(pDev /*CUdeviceptr **/, pHost /*void **/,
                            u /*unsigned int*/);
  // End
}
