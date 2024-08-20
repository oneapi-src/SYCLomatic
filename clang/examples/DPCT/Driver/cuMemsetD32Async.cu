void test(CUdeviceptr d, unsigned u, size_t s, CUstream cs) {
  // Start
  cuMemsetD32Async(d /*CUdeviceptr*/, u /*unsigned*/, s /*size_t*/,
                   cs /*CUstream*/);
  // End
}
