void test(CUdeviceptr pd, const void *pv, size_t s) {
  // Start
  cuMemcpyHtoD(pd /*CUdeviceptr*/, pv /*const void **/, s /*size_t*/);
  // End
}
