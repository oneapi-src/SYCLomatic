void test(void *pv, CUarray a, size_t s1, size_t s2) {
  // Start
  cuMemcpyAtoH(pv /*void **/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
  // End
}
