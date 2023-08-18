void test(void *pv, CUdeviceptr pd, size_t s) {
  // Start
  cuMemcpyDtoH(pv /*void **/, pd /*CUdeviceptr*/, s /*size_t*/);
  // End
}
