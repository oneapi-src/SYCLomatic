void test(CUdeviceptr d, unsigned short us, size_t s, CUstream cs) {
  // Start
  cuMemsetD16Async(d /*CUdeviceptr*/, us /*unsigned short*/, s /*size_t*/,
                   cs /*CUstream*/);
  // End
}
