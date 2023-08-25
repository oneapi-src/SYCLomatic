void test(CUdeviceptr pd, size_t s, CUdevice d) {
  // Start
  CUstream cs;
  cuMemPrefetchAsync(pd /*CUdeviceptr*/, s /*size_t*/, d /*CUdevice*/,
                     cs /*CUstream*/);
  // End
}
