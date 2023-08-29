void test(void *pv, size_t s) {
  // Start
  CUdeviceptr pd;
  CUstream cs;
  cuMemcpyDtoHAsync(pv /*void **/, pd, s /*size_t*/, cs);
  // End
}
