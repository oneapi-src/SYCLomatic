void test(CUdeviceptr d, size_t s1, size_t s2) {
  // Start
  CUarray a;
  cuMemcpyAtoD(d /*CUdeviceptr*/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
  // End
}
