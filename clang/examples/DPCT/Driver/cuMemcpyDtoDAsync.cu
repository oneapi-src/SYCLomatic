void test(CUdeviceptr pd1, CUdeviceptr pd2, size_t s, CUstream cs) {
  // Start
  cuMemcpyDtoDAsync(pd1 /*CUdeviceptr*/, pd2 /*CUdeviceptr*/, s /*size_t*/,
                    cs /*CUstream*/);
  // End
}
