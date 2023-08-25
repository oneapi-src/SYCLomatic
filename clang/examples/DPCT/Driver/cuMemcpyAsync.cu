void test(CUdeviceptr d1, CUdeviceptr d2, size_t s, CUstream cs) {
  // Start
  cuMemcpyAsync(d1 /*CUdeviceptr*/, d2 /*CUdeviceptr*/, s /*size_t*/,
                cs /*CUstream*/);
  // End
}
