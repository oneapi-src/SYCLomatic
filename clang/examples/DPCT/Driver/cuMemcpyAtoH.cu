void test(void *pHost, size_t s1, size_t s2) {
  // Start
  CUarray a;
  cuMemcpyAtoH(pHost /*void **/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
  // End
}
