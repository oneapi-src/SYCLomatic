void test(void *pv, size_t s1, size_t s2) {
  // Start
  CUarray a;
  cuMemcpyAtoH(pv /*void **/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
  // End
}
