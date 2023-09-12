void test(CUdeviceptr d1, CUdeviceptr d2, size_t s) {
  // Start
  CUstream cs;
  cuMemcpyAsync(d1 /*CUdeviceptr*/, d2 /*CUdeviceptr*/, s /*size_t*/,
                cs /*CUstream*/);
  // End
}
