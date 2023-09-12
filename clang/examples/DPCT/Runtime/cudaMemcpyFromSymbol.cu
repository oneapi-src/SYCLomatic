void test(void *dst, const void *symbol, size_t s1, size_t s2) {
  // Start
  cudaMemcpyKind m;
  cudaMemcpyFromSymbol(dst /*void **/, symbol /*const void **/, s1 /*size_t*/,
                       s2 /*size_t*/, m);
  // End
}
