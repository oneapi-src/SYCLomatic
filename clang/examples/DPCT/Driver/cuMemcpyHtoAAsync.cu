void test(CUarray a, size_t s1, const void *pv, size_t s2, CUstream s) {
  // Start
  cuMemcpyHtoAAsync(a /*CUarray*/, s1 /*size_t*/, pv /*const void **/,
                    s2 /*size_t*/, s /*CUstream*/);
  // End
}
