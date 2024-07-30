void test(CUdeviceptr d, unsigned char uc, size_t s, CUstream cs) {
  // Start
  cuMemsetD8Async(d /*CUdeviceptr*/, uc /*unsigned char*/, s /*size_t*/,
                  cs /*CUstream*/);
  // End
}
