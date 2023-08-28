void test(const void *cpv1, const void *cpv2, size_t s1, size_t s2,
          cudaMemcpyKind m) {
  // Start
  cudaMemcpyToSymbol(cpv1 /*const void **/, cpv2 /*const void **/,
                     s1 /*size_t*/, s2 /*size_t*/, m /*cudaMemcpyKind*/);
  // End
}
