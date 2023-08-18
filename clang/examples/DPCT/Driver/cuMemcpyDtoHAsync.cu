void test(void *pv, CUdeviceptr pd, size_t s, CUstream cs) {
  // Start
  cuMemcpyDtoHAsync(pv /*void **/, pd /*CUdeviceptr*/, s /*size_t*/,
                    cs /*CUstream*/);
  // End
}
