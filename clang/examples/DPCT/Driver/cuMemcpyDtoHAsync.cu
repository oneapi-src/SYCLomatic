void test(void *pHost, size_t s) {
  // Start
  CUdeviceptr pDev;
  CUstream stream;
  cuMemcpyDtoHAsync(pHost /*void **/, pDev, s /*size_t*/, stream);
  // End
}
