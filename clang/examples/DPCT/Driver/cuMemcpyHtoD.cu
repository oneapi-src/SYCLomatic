void test(CUdeviceptr pDev, const void *pHost, size_t s) {
  // Start
  cuMemcpyHtoD(pDev /*CUdeviceptr*/, pHost /*const void **/, s /*size_t*/);
  // End
}
