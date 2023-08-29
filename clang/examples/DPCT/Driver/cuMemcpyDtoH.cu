void test(void *pv, size_t s) {
  // Start
  CUdeviceptr pd;
  cuMemcpyDtoH(pv /*void **/, pd, s /*size_t*/);
  // End
}
