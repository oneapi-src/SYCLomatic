void test(void *pv, CUarray a, size_t s1, size_t s2, CUstream s) {
  // Start
  cuMemcpyAtoHAsync(pv /*void **/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/,
                    s /*CUstream*/);
  // End
}
