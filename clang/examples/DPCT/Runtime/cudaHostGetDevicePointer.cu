void test(void **pDev, void *pHost, unsigned int u) {
  // Start
  cudaHostGetDevicePointer(pDev /*void ***/, pHost /*void **/,
                           u /*unsigned int*/);
  // End
}
