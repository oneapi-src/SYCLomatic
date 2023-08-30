void test(void *pHost, size_t s) {
  // Start
  CUdeviceptr pDev;
  cuMemcpyDtoH(pHost /*void **/, pDev, s /*size_t*/);
  // End
}
