void test(CUdeviceptr pd, size_t s, CUdevice d, CUstream cs) {
  // Start
  cuMemPrefetchAsync(pd /*CUdeviceptr*/, s /*size_t*/, d /*CUdevice*/,
                     cs /*CUstream*/);
  // End
}
