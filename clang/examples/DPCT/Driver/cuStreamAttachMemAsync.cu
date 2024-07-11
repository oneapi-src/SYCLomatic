void test(CUstream cs, CUdeviceptr d, size_t s, unsigned int u) {
  // Start
  cuStreamAttachMemAsync(cs /*CUstream*/, d /*CUdeviceptr*/, s /*size_t*/,
                         u /*unsigned int*/);
  // End
}
