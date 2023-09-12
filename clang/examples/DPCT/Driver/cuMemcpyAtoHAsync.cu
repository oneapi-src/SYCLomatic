void test(void *pHost, size_t s1, size_t s2, CUstream s) {
  // Start
  CUarray a;
  cuMemcpyAtoHAsync(pHost /*void **/, a /*CUarray*/, s1 /*size_t*/,
                    s2 /*size_t*/, s /*CUstream*/);
  // End
}
