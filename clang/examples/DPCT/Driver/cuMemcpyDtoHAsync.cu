void test(void *pv, CUdeviceptr pd, size_t s) {
  // Start
  CUstream cs;
  cuMemcpyDtoHAsync(pv /*void **/, pd /*CUdeviceptr*/, s /*size_t*/,
                    cs /*CUstream*/);
  // End
}
