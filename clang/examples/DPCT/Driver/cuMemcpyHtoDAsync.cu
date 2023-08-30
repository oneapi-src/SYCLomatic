void test(CUdeviceptr pd, const void *pv, size_t s) {
  // Start
  CUstream cs;
  cuMemcpyHtoDAsync(pd /*CUdeviceptr*/, pv /*const void **/, s /*size_t*/,
                    cs /*CUstream*/);
  // End
}
