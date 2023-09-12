void test(void *dst, const void *symbol, size_t s1, size_t s2,
          cudaMemcpyKind m) {
  // Start
  cudaStream_t s;
  cudaMemcpyFromSymbolAsync(dst /*void **/, symbol /*const void **/,
                            s1 /*size_t*/, s2 /*size_t*/, m /*cudaMemcpyKind*/,
                            s);
  // End
}
