void test(void **ppv, void *pv, unsigned int u) {
  // Start
  cudaHostGetDevicePointer(ppv /*void ***/, pv /*void **/, u /*unsigned int*/);
  // End
}
