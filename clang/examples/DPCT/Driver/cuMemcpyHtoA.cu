void test(CUarray a, size_t s1, const void *pv, size_t s2) {
  // Start
  cuMemcpyHtoA(a /*CUarray*/, s1 /*size_t*/, pv /*const void **/,
               s2 /*size_t*/);
  // End
}
