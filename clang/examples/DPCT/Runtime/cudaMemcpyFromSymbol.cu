void test(void *pv, const void *cpv, size_t s1, size_t s2, cudaMemcpyKind m) {
  // Start
  cudaMemcpyFromSymbol(pv /*void **/, cpv /*const void **/, s1 /*size_t*/,
                       s2 /*size_t*/, m /*cudaMemcpyKind*/);
  // End
}
