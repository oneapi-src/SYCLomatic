void test(CUarray a, size_t s1, CUdeviceptr d, size_t s2) {
  // Start
  cuMemcpyDtoA(a /*CUarray*/, s1 /*size_t*/, d /*CUdeviceptr*/, s2 /*size_t*/);
  // End
}
