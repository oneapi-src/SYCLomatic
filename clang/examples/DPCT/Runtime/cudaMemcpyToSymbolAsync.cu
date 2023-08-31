void test(const void *symbol, const void *src, size_t s1, size_t s2,
          cudaMemcpyKind m) {
  // Start
  cudaStream_t s;
  cudaMemcpyToSymbolAsync(symbol /*const void **/, src /*const void **/,
                          s1 /*size_t*/, s2 /*size_t*/, m /*cudaMemcpyKind*/,
                          s);
  // End
}
