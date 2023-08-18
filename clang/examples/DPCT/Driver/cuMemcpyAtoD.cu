void test(CUdeviceptr d, CUarray a, size_t s1, size_t s2) {
  // Start
  cuMemcpyAtoD(d /*CUdeviceptr*/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
  // End
}
