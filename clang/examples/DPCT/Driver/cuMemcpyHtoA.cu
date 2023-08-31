void test(size_t s1, const void *pHost, size_t s2) {
  // Start
  CUarray a;
  cuMemcpyHtoA(a /*CUarray*/, s1 /*size_t*/, pHost /*const void **/,
               s2 /*size_t*/);
  // End
}
